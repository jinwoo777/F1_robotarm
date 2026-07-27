"""shake_test_node — 파지 직후 안정성 검증(Shake Test) 노드.

설계 문서 "예외처리 1+. 불완전 파지(Grip Unstable)"에 대응:
  GRIP 성공(/gripper_state == 'GRIPPED') 확인 → 소폭 리프트 + 좌우 흔들기 →
  흔들기 동안의 힘/토크 변화와 그리퍼 파지 유지 여부로 안정성 판정 →
    안정  : 아무것도 하지 않음 (이미 GRIPPED 상태이므로 다음 단계로 진행)
    불안정: /gripper_state = 'GRIP_UNSTABLE' 발행
            → safety_monitor가 이를 받아 /alarm + /estop 발행
            → recovery_manager가 HOME 복귀 후 재파지 재시도 (최대 3회)

★★★ 실기 배포 전 필수 확인 사항 ★★★
아래 LIFT_MM / WIGGLE_DEG / WIGGLE_AXIS / STABLE_* 값은 전부 placeholder다.
이 노드는 절대 자세(posx)를 새로 지정하지 않고 "GRIP 시점의 현재 자세 기준 상대이동"만 사용해서
임의의 좌표를 잘못 넣어 충돌하는 위험은 피했지만, 리프트/흔들기 폭과 안정성 판정 임계값은
실로봇에서 실제 웍 손잡이를 잡은 상태로 측정해보기 전에는 안전/정확성을 보장할 수 없다.
wok_data_logger로 정상 사이클 힘/토크 데이터를 먼저 수집해 STABLE_* 값을 보정할 것을 권장한다.
"""
import threading
from collections import deque

import rclpy
import DR_init
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Wrench

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 20, 20        # 흔들기 테스트용 저속 (원래 이동 속도보다 느리게)

# --- placeholder: 실로봇에서 확인 후 조정 ---
LIFT_MM = 20.0                 # 리프트 높이 (mm), 툴 Z축 기준 상대이동이라고 가정
WIGGLE_DEG = 5.0                # 흔들기 각도 (deg)
WIGGLE_AXIS_INDEX = 3           # move_periodic amp의 축 인덱스 (0..5 = x,y,z,rx,ry,rz). 웍 좌우 흔들기에 맞는 축으로 조정 필요
WIGGLE_PERIOD_S = 1.0
WIGGLE_REPEAT = 2
STABLE_MY_DEVIATION = 4.0       # 흔들기 중 My 변동 허용치(Nm) — 이 이상 튀면 GRIP_UNSTABLE
STABLE_FZ_DEVIATION = 10.0      # 흔들기 중 Fz 변동 허용치(N)
# ---------------------------------------

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class ShakeTestNode(Node):
    def __init__(self):
        super().__init__('shake_test_node', namespace=ROBOT_ID)

        setattr(DR_init, '__dsr__node', self)
        try:
            from DSR_ROBOT2 import movel, move_periodic, DR_TOOL, DR_MV_MOD_REL
            from DR_common2 import posx
        except ImportError as e:
            self.get_logger().error(f'DSR_ROBOT2 import 실패: {e}')
            raise

        self._movel = movel
        self._move_periodic = move_periodic
        self._DR_TOOL = DR_TOOL
        self._DR_MV_MOD_REL = DR_MV_MOD_REL
        self._posx = posx

        self.gripper_state_sub = self.create_subscription(
            String, '/gripper_state', self.gripper_state_callback, 10
        )
        self.force_feedback_sub = self.create_subscription(
            Wrench, '/force_feedback', self.force_feedback_callback, 10
        )
        self.gripper_state_pub = self.create_publisher(String, '/gripper_state', 10)

        self._tested_this_grip = False
        self._testing = False
        self._wrench_buffer = deque(maxlen=200)

        self.get_logger().info('shake_test_node 시작 — GRIPPED 확인 후 Shake Test 수행')

    def force_feedback_callback(self, msg: Wrench):
        self._wrench_buffer.append((msg.torque.y, msg.force.z))

    def gripper_state_callback(self, msg):
        state = msg.data
        if state == 'GRIPPED' and not self._tested_this_grip and not self._testing:
            self._tested_this_grip = True
            threading.Thread(target=self._run_shake_test, daemon=True).start()
        elif state != 'GRIPPED':
            # 다음에 새로 파지했을 때 다시 테스트하도록 리셋
            self._tested_this_grip = False

    def _run_shake_test(self):
        self._testing = True
        self.get_logger().info('Shake Test 시작 — 리프트 + 흔들기')

        self._wrench_buffer.clear()
        try:
            lift = self._posx([0.0, 0.0, LIFT_MM, 0.0, 0.0, 0.0])
            self._movel(lift, vel=VELOCITY, acc=ACC, ref=self._DR_TOOL, mod=self._DR_MV_MOD_REL)

            amp = [0.0] * 6
            amp[WIGGLE_AXIS_INDEX] = WIGGLE_DEG
            self._move_periodic(
                amp=amp,
                period=[WIGGLE_PERIOD_S] * 6,
                atime=0.2,
                repeat=WIGGLE_REPEAT,
                ref=self._DR_TOOL,
            )
        except Exception as e:
            self.get_logger().error(f'Shake Test 모션 중 오류: {e} — 안전을 위해 GRIP_UNSTABLE로 판정')
            self._publish_unstable()
            self._testing = False
            return

        stable = self._evaluate_stability()
        if stable:
            self.get_logger().info('Shake Test 통과 — 파지 안정적, GRIPPED 유지')
        else:
            self.get_logger().warn('Shake Test 실패 — 파지 불안정으로 판정')
            self._publish_unstable()

        self._testing = False

    def _evaluate_stability(self):
        samples = list(self._wrench_buffer)
        if not samples:
            self.get_logger().warn('Shake Test 중 /force_feedback 샘플을 받지 못함 — 안전을 위해 불안정으로 판정')
            return False

        my_values = [s[0] for s in samples]
        fz_values = [s[1] for s in samples]
        my_swing = max(my_values) - min(my_values)
        fz_swing = max(fz_values) - min(fz_values)

        self.get_logger().info(
            f'Shake Test 변동폭: My={my_swing:.2f}Nm(허용 {STABLE_MY_DEVIATION}), '
            f'Fz={fz_swing:.2f}N(허용 {STABLE_FZ_DEVIATION})'
        )
        return my_swing <= STABLE_MY_DEVIATION and fz_swing <= STABLE_FZ_DEVIATION

    def _publish_unstable(self):
        self.gripper_state_pub.publish(String(data='GRIP_UNSTABLE'))


def main(args=None):
    rclpy.init(args=args)
    node = ShakeTestNode()
    # Shake Test는 별도 스레드에서 movel/move_periodic을 호출하는데, 그 안에서 다시
    # 같은 노드를 spin_until_future_complete로 재진입 spin한다 — MultiThreadedExecutor로 풀어줌.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
