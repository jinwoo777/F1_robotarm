from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Empty
from geometry_msgs.msg import Wrench

# 하드 임계값 (넘으면 즉시 /estop) — 정상 웍질 My(~4Nm)보다 충분히 높게 설정
MY_LIMIT = 15.0
FZ_LIMIT = 40.0

# Rule-base 소프트 경보 임계값 — 하드 임계값보다 낮은, "최근 정상 흐름" 대비 급격한 이탈 감지용
# (AI/Isolation Forest 대신 이동평균 기반 규칙으로 대체)
BASELINE_WINDOW = 50          # 정상 흐름 기준으로 삼을 최근 샘플 개수
FLOW_DEVIATION_MY = 6.0       # 이동평균 대비 이 이상 벗어나면 소프트 경보
FLOW_DEVIATION_FZ = 15.0
MIN_BASELINE_SAMPLES = 20     # 이만큼 쌓이기 전에는 판단하지 않음 (초기 노이즈 방지)


class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Subscribers
        self.gripper_state_sub = self.create_subscription(
            String,
            '/gripper_state',
            self.gripper_state_callback,
            10
        )

        self.force_feedback_sub = self.create_subscription(
            Wrench,
            '/force_feedback',
            self.force_feedback_callback,
            10
        )

        self.reset_estop_sub = self.create_subscription(
            Empty,
            '/reset_estop',
            self.reset_estop_callback,
            10
        )

        self.calibrate_sub = self.create_subscription(
            Empty,
            '/calibrate',
            self.calibrate_callback,
            10
        )

        # Publishers
        self.alarm_pub = self.create_publisher(String, '/alarm', 10)
        self.estop_pub = self.create_publisher(Bool, '/estop', 10)

        # 파지 상실(웍질 중 이탈) 판정을 위한 이전 상태 추적
        self._last_gripper_state = None

        # Rule-base 이상흐름 판정을 위한 정상 샘플 이동평균 버퍼
        self._my_history = deque(maxlen=BASELINE_WINDOW)
        self._fz_history = deque(maxlen=BASELINE_WINDOW)

        self.get_logger().info('Safety Monitor Node has been started.')

    def gripper_state_callback(self, msg):
        state = msg.data
        prev_state = self._last_gripper_state

        if state in ['GRIP_FAILED', 'GRIP_UNSTABLE']:
            self.get_logger().warn(f'Detected abnormal gripper state: {state}')
            self.trigger_alarm(state)
        elif prev_state == 'GRIPPED' and state in ['NOT_GRIPPED', 'GRIP_LOST']:
            # 웍질 중 정상 파지 상태였다가 놓친 경우 — 초기 파지 실패(GRIP_FAILED)와는 별개 코드로 구분
            self.get_logger().warn(f'Pan dropped during work: {prev_state} -> {state}')
            self.trigger_alarm('PAN_DROPPED')

        self._last_gripper_state = state

    def force_feedback_callback(self, msg):
        my = msg.torque.y
        fz = msg.force.z

        # 1) 하드 임계값 — 사람 접촉/충돌 의심, 즉시 정지
        if abs(my) > MY_LIMIT or abs(fz) > FZ_LIMIT:
            self.get_logger().warn(f'Human contact detected! My: {my:.2f} Nm, Fz: {fz:.2f} N')
            self.trigger_alarm('HUMAN_CONTACT')
            return

        # 2) Rule-base 소프트 경보 — 최근 정상 흐름(이동평균) 대비 급격한 이탈만 검사, estop 없이 경보만
        if len(self._my_history) >= MIN_BASELINE_SAMPLES:
            baseline_my = sum(self._my_history) / len(self._my_history)
            baseline_fz = sum(self._fz_history) / len(self._fz_history)
            dev_my = abs(my - baseline_my)
            dev_fz = abs(fz - baseline_fz)
            if dev_my > FLOW_DEVIATION_MY or dev_fz > FLOW_DEVIATION_FZ:
                self.raise_flow_anomaly(
                    f'My {my:.2f}Nm(기준 {baseline_my:.2f}) / Fz {fz:.2f}N(기준 {baseline_fz:.2f}) — 정상 흐름 이탈'
                )
                # 이상 샘플은 기준선에 반영하지 않음 (기준이 오염되는 것 방지)
                return

        self._my_history.append(my)
        self._fz_history.append(fz)

    def reset_estop_callback(self, msg):
        # recovery_manager(자동복구) / estop_button_io(DI16 재개)가 원인 해소를 확인한 뒤 호출.
        # 이 노드가 /estop 발행 주체이므로, 실제로 해제 신호(False)를 여기서 내려줘야 다른 노드들이
        # E-STOP이 풀렸다는 걸 알 수 있다 (지금까지는 /reset_estop을 받아줄 곳이 없어 영구 래치되던 문제).
        self.estop_pub.publish(Bool(data=False))
        self.get_logger().info('/reset_estop 수신 — E-STOP 해제(/estop=False) 발행')

    def calibrate_callback(self, msg):
        # 실물 힘/토크 센서를 다시 0점 잡는 API는 없음(M0609은 관절 토크로 힘을 추정하는 방식).
        # 대신 HOME 복귀 후에는 위치가 바뀌었으니, rule-base 이상탐지가 쓰는 "최근 정상 흐름" 기준선을
        # 리셋해서 새 위치 기준으로 다시 학습하게 한다.
        self._my_history.clear()
        self._fz_history.clear()
        self.get_logger().info('/calibrate 수신 — FLOW_ANOMALY 기준선(이동평균) 초기화')

    def raise_flow_anomaly(self, message):
        # 하드 estop 없이 경보만 발행하는 소프트 경보 — recovery_manager는 이 코드에 자동조치를 하지 않음
        alarm_msg = String()
        alarm_msg.data = 'FLOW_ANOMALY'
        self.alarm_pub.publish(alarm_msg)
        self.get_logger().warn(f'[FLOW_ANOMALY] {message}')

    def trigger_alarm(self, alarm_type):
        alarm_msg = String()
        alarm_msg.data = alarm_type
        self.alarm_pub.publish(alarm_msg)

        estop_msg = Bool()
        estop_msg.data = True
        self.estop_pub.publish(estop_msg)

        self.get_logger().info(f'Published Alarm: {alarm_type} and Triggered E-STOP.')

def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
