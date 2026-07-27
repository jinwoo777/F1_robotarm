"""robot_state_watchdog — 로봇 자체 안전정지(Safety Stop) 상태를 직접 감지.

힘/토크 임계값(force_feedback_bridge + safety_monitor의 HUMAN_CONTACT 판정)은
사람이 미는 힘이 우리 소프트웨어 문턱(15Nm/40N)을 넘어야만 잡히는데, 실제로는
로봇 자체 하드웨어 안전기능이 훨씬 예민해서 그게 먼저 STATE_SAFE_STOP으로 넘어가
버리는 경우가 많다. 이 노드는 힘 값을 보는 대신 로봇의 실제 상태(get_robot_state())를
직접 폴링해서, STATE_SAFE_STOP류로 전이되는 순간 바로 /alarm + /estop을 발행한다.

★ DSR_ROBOT2.get_robot_state()를 안 쓰고 dsr_msgs2 서비스를 직접 완전 논블로킹으로 호출한다.
  DSR_ROBOT2.py의 get_robot_state()는 rclpy.spin_until_future_complete(g_node, future)를
  타임아웃 없이 호출해서, dsr_controller2가 진행 중인 movel/movej 처리로 바빠 응답이
  늦으면 이 폴링 자체가 무한정 멈춰버린다 — 그 사이 실제로 안전정지가 걸려도 한참 뒤에야
  (밀린 응답이 오고 나서야) 감지되는 지연이 발생했었다. 여기선 _poll() 안에서 절대 대기하지
  않고, 이전 요청의 완료 여부만 매 틱(0.2초)마다 확인한다 — 응답이 늦어도 폴링 자체는
  안 막히므로 최소 0.2초 주기로 계속 확인할 수 있다.

  ros2 run wok_exception_handling robot_state_watchdog
"""
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool
from dsr_msgs2.srv import GetRobotState

ROBOT_ID = "dsr01"
POLL_HZ = 5.0

# DRFC.py 기준
STATE_SAFE_OFF = 3
STATE_SAFE_STOP = 5
STATE_EMERGENCY_STOP = 6
STATE_SAFE_STOP2 = 9
STATE_SAFE_OFF2 = 10
TRIGGER_STATES = {STATE_SAFE_OFF, STATE_SAFE_STOP, STATE_EMERGENCY_STOP, STATE_SAFE_STOP2, STATE_SAFE_OFF2}


class RobotStateWatchdog(Node):
    def __init__(self):
        super().__init__('robot_state_watchdog', namespace=ROBOT_ID)

        self._GetRobotState = GetRobotState
        self._get_robot_state_cli = self.create_client(GetRobotState, 'system/get_robot_state')
        self._pending_future = None   # 이전 폴이 아직 응답 안 왔으면 재요청 대신 계속 기다림

        self.alarm_pub = self.create_publisher(String, '/alarm', 10)
        self.estop_pub = self.create_publisher(Bool, '/estop', 10)

        self._prev_state = None
        self._tripped = False   # 같은 안전정지 에피소드에서 알람 중복 발행 방지

        cb = ReentrantCallbackGroup()
        self.create_timer(1.0 / POLL_HZ, self._poll, callback_group=cb)

        self.get_logger().info(f'robot_state_watchdog 시작 — {POLL_HZ:.0f}Hz로 robot_state 감시')

    def _poll(self):
        """블로킹 없이 매 틱마다 호출됨. 이전 요청이 아직 응답 안 왔으면 그냥 넘어가고,
        응답이 왔으면 처리한 뒤 다음 요청을 낸다 — 한 번에 요청 하나만 미해결로 둬서
        dsr_controller2가 바쁠 때 요청이 쌓이는 것도 방지한다."""
        if self._pending_future is not None:
            if not self._pending_future.done():
                return  # 아직 응답 안 옴 — 절대 여기서 안 기다리고 다음 틱에 다시 확인
            future, self._pending_future = self._pending_future, None
            result = future.result()
            if result is not None and result.success:
                self._handle_state(result.robot_state)

        req = self._GetRobotState.Request()
        self._pending_future = self._get_robot_state_cli.call_async(req)

    def _handle_state(self, state):
        if state in TRIGGER_STATES:
            if not self._tripped:
                self._tripped = True
                self.get_logger().warn(f'로봇 안전정지 상태 감지 (robot_state={state}) — HUMAN_CONTACT 알람 발행')
                self.alarm_pub.publish(String(data='HUMAN_CONTACT'))
                self.estop_pub.publish(Bool(data=True))
        else:
            # 정상 상태로 돌아오면 다음 에피소드를 다시 감지할 수 있도록 리셋
            self._tripped = False

        self._prev_state = state


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateWatchdog()
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
