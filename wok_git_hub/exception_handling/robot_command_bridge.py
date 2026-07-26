"""robot_command_bridge — recovery_manager의 /reset_robot 명령을 실제 로봇 동작으로 실행.

지금까지 recovery_manager는 /reset_robot(String 'HOME')을 발행만 하고, 이걸 받아서
실제로 로봇을 홈으로 보내는 노드가 없어서 자동복구가 로그만 찍고 아무 일도 안 했다.
HOME 자세는 rokey의 다른 스크립트들(move.py, m0609_gear_force.py, wok_test4.py)이
공통으로 쓰는 posj(0,0,90,0,90,0)과 동일하게 맞췄다 — 새 좌표를 임의로 만들지 않음.

  ros2 run wok_exception_handling robot_command_bridge
"""
import rclpy
import DR_init
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 30, 30

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class RobotCommandBridge(Node):
    def __init__(self):
        super().__init__('robot_command_bridge', namespace=ROBOT_ID)

        setattr(DR_init, '__dsr__node', self)
        try:
            from DSR_ROBOT2 import movej
            from DR_common2 import posj
            from dsr_msgs2.srv import SetRobotControl
        except ImportError as e:
            self.get_logger().error(f'DSR_ROBOT2 import 실패: {e}')
            raise

        self._movej = movej
        self._homej = posj([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self._busy = False
        # main()에서 채워짐. _clear_safe_stop()의 rclpy.spin_until_future_complete가 이 executor를
        # '명시적으로' 지정해야 한다 — 인자 없이 부르면 암묵적으로 프로세스 전역 executor를 쓰는데,
        # /reset_robot 콜백은 아래 ReentrantCallbackGroup 덕분에 main()의 MultiThreadedExecutor
        # 스레드 풀에서 실행된다. 거기서 또 다른(전역) executor로 중첩 spin하면 같은 노드를 서로
        # 다른 executor가 동시에 건드리게 돼 rclpy/rcl 레벨에서 wait set이 깨질 수 있다
        # (다른 스크립트에서 실제로 이 패턴 때문에 `IndexError: wait set index too big`로
        # 프로세스가 죽는 걸 확인했다 — 별도 스레드를 수동으로 만들어 그 안에서 spin하는 것도
        # 마찬가지 문제라 여기선 아예 수동 threading.Thread를 쓰지 않고, 콜백 자체가
        # ReentrantCallbackGroup을 통해 executor의 스레드풀에서 돌게 한다).
        self.executor = None

        # DSR_ROBOT2.py 파이썬 래퍼엔 set_robot_control이 없어서 dsr_msgs2 서비스를 직접 호출.
        # CONTROL_RESET_SAFET_STOP=2 : STATE_SAFE_STOP -> STATE_STANDBY로 전환.
        # 이게 없으면 로봇이 실제 안전정지 상태로 남아있어서 movej 자체가 거부된다.
        self._CONTROL_RESET_SAFET_STOP = 2
        self._set_robot_control_cli = self.create_client(SetRobotControl, 'system/set_robot_control')
        self._SetRobotControl = SetRobotControl

        # ReentrantCallbackGroup: 이 콜백 안에서 movej 등이 블로킹(spin_until_future_complete)돼도
        # executor의 다른 스레드가 계속 다른 콜백/서비스 응답을 처리할 수 있다.
        cb = ReentrantCallbackGroup()
        self.create_subscription(String, '/reset_robot', self.reset_robot_callback, 10, callback_group=cb)
        self.get_logger().info(
            'robot_command_bridge 시작 — /reset_robot: "HOME"=안전정지 해제+홈 복귀, "CLEAR"=안전정지 해제만'
        )

    def reset_robot_callback(self, msg):
        # "HOME" : 안전정지 해제 후 홈 자세로 이동 (재료투입1처럼 홈에서 시작하는 복구)
        # "CLEAR": 안전정지만 해제하고 로봇은 멈춘 자리에 그대로 둠
        #          (재료투입2 이후 단계 — 홈 복귀 없이 그 단계 시작위치로 wok_test4가 직접 이동)
        # 별도 스레드를 수동으로 만들지 않고 콜백 자체에서 바로 처리한다 — ReentrantCallbackGroup +
        # MultiThreadedExecutor 덕분에 이 콜백이 블로킹돼도 executor의 다른 워커가 계속 돈다.
        if msg.data not in ('HOME', 'CLEAR'):
            return
        if self._busy:
            self.get_logger().warn('이미 복구 동작 중 — 중복 요청 무시')
            return
        self._recover(msg.data == 'HOME')

    def _recover(self, go_home):
        self._busy = True
        try:
            self._clear_safe_stop()
            if go_home:
                self.get_logger().info('HOME 복귀 시작')
                self._movej(self._homej, vel=VELOCITY, acc=ACC)
                self.get_logger().info('HOME 복귀 완료')
            else:
                self.get_logger().info('안전정지만 해제 — 홈 복귀 없이 현재 위치 유지')
        except Exception as e:
            self.get_logger().error(f'복구 동작 중 오류: {e}')
        finally:
            self._busy = False

    def _clear_safe_stop(self):
        # 로봇이 실제 STATE_SAFE_STOP이면 movej가 거부되므로 먼저 풀어준다.
        # 안전정지 상태가 아니었을 때 호출해도(무해 — 그냥 success=False로 응답) 문제 없다.
        if not self._set_robot_control_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('system/set_robot_control 서비스를 찾을 수 없음 — 안전정지 해제 건너뜀')
            return
        req = self._SetRobotControl.Request()
        req.robot_control = self._CONTROL_RESET_SAFET_STOP
        future = self._set_robot_control_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, executor=self.executor, timeout_sec=2.0)
        result = future.result()
        if result and result.success:
            self.get_logger().info('로봇 안전정지 해제 완료 (STATE_SAFE_STOP -> STANDBY)')
        else:
            self.get_logger().info('안전정지 해제 응답 없음/실패 — 이미 정상 상태였을 수 있음')


def main(args=None):
    rclpy.init(args=args)
    node = RobotCommandBridge()
    # /reset_robot 콜백(ReentrantCallbackGroup)이 블로킹돼도 다른 콜백이 계속 돌도록
    # MultiThreadedExecutor 사용. node.executor에 저장해 _clear_safe_stop()이 명시적으로
    # 같은 executor를 재사용하게 한다(암묵적 전역 executor와의 충돌 방지).
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    node.executor = executor
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
