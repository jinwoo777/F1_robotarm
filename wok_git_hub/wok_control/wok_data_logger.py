"""wok_data_logger — 로봇 6축(joint) 각도와 그리퍼(OnRobot RG2) 폭을 CSV로 기록.

wok_test4.py 와 별도 프로세스로 동시에 실행한다 (동일 프로세스에서 돌리면
wok_test4 의 movel/movej 블로킹 호출 때문에 주기적인 로깅이 밀릴 수 있음).

  터미널 1: ros2 run rokey wok_test4
  터미널 2: ros2 run rokey wok_data_logger

동작 확인 순서 (실행 전):
  1. dsr01 드라이버가 떠 있어야 get_current_posj() 서비스 호출이 됨.
  2. OnRobot RG2 컨트롤러(onrobot_rg_control)가 떠 있어야 /joint_states, /onrobot/pose
     로 그리퍼 폭을 뽑을 수 있음. 안 떠 있으면 gripper_width_mm 칸은 빈 값으로 기록됨.
"""
import csv
import time

import rclpy

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

CSV_PATH = "wok_data_log.csv"   # 필요하면 절대경로로 바꿔서 사용
SAMPLE_HZ = 10.0                # 로깅 주기 (초당 샘플 수)

GRIPPER_JOINT_NAME = "finger_joint"   # OnRobotRGControllerServer 가 publish 하는 /joint_states 의 대표 조인트명
GRIPPER_POSE_SERVICE = "/onrobot/pose"

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("wok_data_logger", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import get_current_posj
    from sensor_msgs.msg import JointState
    from onrobot_rg_msgs.srv import GripperPose

    logger = node.get_logger()

    gripper_theta = {"value": None}

    def on_joint_states(msg):
        if GRIPPER_JOINT_NAME in msg.name:
            idx = msg.name.index(GRIPPER_JOINT_NAME)
            gripper_theta["value"] = msg.position[idx]

    node.create_subscription(JointState, "/joint_states", on_joint_states, 10)
    pose_client = node.create_client(GripperPose, GRIPPER_POSE_SERVICE)

    if not pose_client.wait_for_service(timeout_sec=3.0):
        logger.warn(
            "%s 서비스를 찾을 수 없습니다. OnRobot RG2 컨트롤러가 켜져 있는지 확인하세요. "
            "그리퍼 폭 없이 6축 데이터만 기록합니다." % GRIPPER_POSE_SERVICE)

    def query_gripper_width_mm():
        theta = gripper_theta["value"]
        if theta is None or not pose_client.service_is_ready():
            return ""
        req = GripperPose.Request()
        req.known.theta = theta
        future = pose_client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=0.2)
        if future.done() and future.result() is not None:
            return future.result().pose.x * 1000.0  # m -> mm
        return ""

    period = 1.0 / SAMPLE_HZ
    logger.info("wok_data_logger 시작 — %s 에 기록, %.1fHz" % (CSV_PATH, SAMPLE_HZ))

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "j1", "j2", "j3", "j4", "j5", "j6", "gripper_width_mm"])

        try:
            while rclpy.ok():
                t0 = time.time()
                rclpy.spin_once(node, timeout_sec=0.0)

                cur_posj = get_current_posj()
                width_mm = query_gripper_width_mm()

                writer.writerow([t0] + list(cur_posj)[:6] + [width_mm])
                f.flush()

                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            pass

    logger.info("wok_data_logger 종료")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
