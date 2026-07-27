"""force_feedback_bridge — 실로봇 힘/토크 값을 /force_feedback(Wrench)으로 발행.

지금까지 safety_monitor의 HUMAN_CONTACT/FLOW_ANOMALY 판정은 /force_feedback을
구독만 하고 있었고, 이걸 실제로 채워주는 노드가 없어서 절대 발동하지 않았다.
이 노드가 DSR_ROBOT2.get_tool_force()로 실측값을 읽어 그 공백을 메운다.

  ros2 run wok_exception_handling force_feedback_bridge
"""
import rclpy
import DR_init
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Wrench

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
PUBLISH_HZ = 10.0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


class ForceFeedbackBridge(Node):
    def __init__(self):
        super().__init__('force_feedback_bridge', namespace=ROBOT_ID)

        setattr(DR_init, '__dsr__node', self)
        try:
            from DSR_ROBOT2 import get_tool_force, DR_BASE
        except ImportError as e:
            self.get_logger().error(f'DSR_ROBOT2 import 실패: {e}')
            raise

        self._get_tool_force = get_tool_force
        self._DR_BASE = DR_BASE

        self.force_feedback_pub = self.create_publisher(Wrench, '/force_feedback', 10)
        # get_tool_force()가 내부적으로 rclpy.spin_until_future_complete(g_node, ...)를 호출하는데,
        # main()의 spin과 같은 콜백그룹이면 재진입이 막혀 영원히 대기하게 된다 — Reentrant로 풀어줌.
        cb = ReentrantCallbackGroup()
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_force, callback_group=cb)

        self.get_logger().info(f'force_feedback_bridge 시작 — {PUBLISH_HZ:.0f}Hz로 /force_feedback 발행')

    def _publish_force(self):
        tool_force = self._get_tool_force(ref=self._DR_BASE)
        # 실패 시 get_tool_force()는 int -1을 반환한다 (list가 아님).
        # not -1 은 False라 예전 체크(`if not tool_force`)를 그냥 통과해버리고
        # len(-1)에서 TypeError가 나던 버그 — list인지 먼저 확인하도록 수정.
        if not isinstance(tool_force, list) or len(tool_force) < 6:
            return

        msg = Wrench()
        msg.force.x, msg.force.y, msg.force.z = tool_force[0], tool_force[1], tool_force[2]
        msg.torque.x, msg.torque.y, msg.torque.z = tool_force[3], tool_force[4], tool_force[5]
        self.force_feedback_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ForceFeedbackBridge()
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
