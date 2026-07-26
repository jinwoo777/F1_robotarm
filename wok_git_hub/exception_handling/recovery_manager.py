import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Empty
import time

class RecoveryManager(Node):
    def __init__(self):
        super().__init__('recovery_manager')

        # Subscribers
        self.alarm_sub = self.create_subscription(
            String,
            '/alarm',
            self.alarm_callback,
            10
        )
        self.estop_sub = self.create_subscription(
            Bool,
            '/estop',
            self.estop_callback,
            10
        )
        self.manual_resume_sub = self.create_subscription(
            Empty,
            '/manual_resume',
            self.manual_resume_callback,
            10
        )

        # Publishers
        self.reset_robot_pub = self.create_publisher(String, '/reset_robot', 10)
        self.calibrate_pub = self.create_publisher(Empty, '/calibrate', 10)
        self.gripper_cmd_pub = self.create_publisher(String, '/gripper_command', 10)
        self.reset_estop_pub = self.create_publisher(Empty, '/reset_estop', 10)
        self.alarm_pub = self.create_publisher(String, '/alarm', 10) # For AUTO_RECOVERY_FAILED

        # State
        self.is_estopped = False
        self.recovery_attempts = {
            'GRIP_FAILED': 0,
            'GRIP_UNSTABLE': 0,
            'PAN_DROPPED': 0
        }
        self.max_attempts = 3

        # 물리 E-STOP 버튼(DI13)으로 정지된 경우: 자동복구 금지, DI16(재개) 입력을 기다림
        self.manual_estop_pending = False

        # 같은 사건에 대해 여러 노드(wok_test4 직접감지 + robot_state_watchdog)가
        # 동시에 HUMAN_CONTACT를 발행해도 복구 시퀀스(HOME 복귀)가 두 번 돌지 않도록 래치.
        self.recovery_in_progress = False

        self.get_logger().info('Recovery Manager Node has been started.')

    def estop_callback(self, msg):
        self.is_estopped = msg.data
        if self.is_estopped:
            self.get_logger().info('E-STOP active. Waiting for recovery sequence.')
        else:
            # 해제되면(복구 완료) 다음 사건을 다시 받을 수 있도록 중복가드 래치 해제
            self.recovery_in_progress = False

    def alarm_callback(self, msg):
        alarm_type = msg.data

        if alarm_type == 'FLOW_ANOMALY':
            # Rule-base 소프트 경보 — estop을 동반하지 않으므로 자동조치 없이 로깅만 (대시보드용)
            self.get_logger().warn(f'[FLOW_ANOMALY] 정상 흐름 이탈 감지 — 자동조치 없음 (경보만)')
            return

        # MANUAL_ESTOP은 래치 처리(자동복구 안 함)이므로 중복가드 밖에서 바로 처리
        if alarm_type == 'MANUAL_ESTOP':
            self.handle_manual_estop()
            return

        # 자동복구 대상 알람: 이미 복구 시퀀스가 진행 중이면 같은 사건 중복으로 보고 무시
        if self.recovery_in_progress:
            self.get_logger().warn(f'복구 진행 중 — 중복 알람({alarm_type}) 무시.')
            return

        self.get_logger().info(f'Received alarm: {alarm_type}. Initiating recovery.')
        # 래치: 복구가 끝나(=/estop 해제) estop_callback에서 풀릴 때까지 유지.
        # 같은 사건에 대한 중복 알람(다른 노드가 동시에 발행)이 큐에 남아 있어도
        # 여기서 걸러져 HOME 복귀가 두 번 도는 것을 막는다.
        self.recovery_in_progress = True
        if alarm_type == 'GRIP_FAILED':
            self.handle_grip_failed()
        elif alarm_type == 'GRIP_UNSTABLE':
            self.handle_grip_unstable()
        elif alarm_type == 'HUMAN_CONTACT':
            self.handle_human_contact()
        elif alarm_type == 'PAN_DROPPED':
            self.handle_pan_dropped()

    def handle_grip_failed(self):
        if self.recovery_attempts['GRIP_FAILED'] >= self.max_attempts:
            self.get_logger().error('Max recovery attempts reached for GRIP_FAILED. Calling Administrator.')
            self.publish_auto_recovery_failed()
            return
            
        self.recovery_attempts['GRIP_FAILED'] += 1
        self.get_logger().info(f"GRIP_FAILED Recovery Attempt: {self.recovery_attempts['GRIP_FAILED']}")
        
        # Recovery sequence — HOME/재파지 시도가 끝난 뒤에 estop을 풀어야
        # wok_test4가 그 전에 먼저 재개해버리는 경합을 피할 수 있다.
        self.reset_robot_to_home()
        self.calibrate_robot()

        # Retry grip
        self.get_logger().info('Retrying grip...')
        grip_msg = String()
        grip_msg.data = 'GRIP'
        self.gripper_cmd_pub.publish(grip_msg)

        self.reset_estop_pub.publish(Empty())

    def handle_grip_unstable(self):
        if self.recovery_attempts['GRIP_UNSTABLE'] >= self.max_attempts:
            self.get_logger().error('Max recovery attempts reached for GRIP_UNSTABLE. Calling Administrator.')
            self.publish_auto_recovery_failed()
            return
            
        self.recovery_attempts['GRIP_UNSTABLE'] += 1
        self.get_logger().info(f"GRIP_UNSTABLE Recovery Attempt: {self.recovery_attempts['GRIP_UNSTABLE']}")
        
        # Recovery sequence
        self.reset_robot_to_home()

        # Retry grip and shake test (assuming wok_controller or arm_driver initiates shake test on GRIP)
        self.get_logger().info('Retrying grip for Shake Test...')
        grip_msg = String()
        grip_msg.data = 'GRIP'
        self.gripper_cmd_pub.publish(grip_msg)

        self.reset_estop_pub.publish(Empty())

    def handle_human_contact(self):
        self.get_logger().info('Waiting 2 seconds for stabilization...')
        time.sleep(2.0)

        # 홈으로 강제 복귀시키지 않는다 — 안전정지만 해제하고 로봇은 멈춘 자리에 그대로 둔다.
        # 어느 위치로 갈지는 wok_test4의 단계 재시작이 결정한다:
        #   - 재료투입1 단계는 첫 동작이 movej(home)이라 자연히 홈으로 가고,
        #   - 재료투입2 이후 단계는 해당 단계의 시작 위치로 바로 이동한다(홈 안 거침).
        # 안전정지 해제가 끝난 뒤에 estop을 풀어야 wok_test4가 먼저 재개해버리는 경합을 피한다.
        self.clear_safe_stop_only()
        self.calibrate_robot()

        self.get_logger().info('Resetting E-STOP.')
        self.reset_estop_pub.publish(Empty())
        self.get_logger().info('Human Contact recovery completed. Resuming work.')

    def handle_pan_dropped(self):
        # 웍질 도중 파지를 놓친 경우 — 초기 파지 실패(GRIP_FAILED)와 카운터를 분리해서 관리
        if self.recovery_attempts['PAN_DROPPED'] >= self.max_attempts:
            self.get_logger().error('Max recovery attempts reached for PAN_DROPPED. Calling Administrator.')
            self.publish_auto_recovery_failed()
            return

        self.recovery_attempts['PAN_DROPPED'] += 1
        self.get_logger().info(f"PAN_DROPPED Recovery Attempt: {self.recovery_attempts['PAN_DROPPED']}")

        # 복구 절차: HOME 복귀 → 재파지 → (성공 시) 중단된 사이클부터 재개
        self.reset_robot_to_home()

        self.get_logger().info('Retrying grip after pan drop...')
        grip_msg = String()
        grip_msg.data = 'GRIP'
        self.gripper_cmd_pub.publish(grip_msg)

        self.reset_estop_pub.publish(Empty())

    def handle_manual_estop(self):
        # 운영자가 물리 E-STOP 버튼(DI13)을 누른 경우 — 자동복구를 절대 시도하지 않고
        # DI16(재개) 버튼 입력(/manual_resume)이 올 때까지 래치 유지
        self.manual_estop_pending = True
        self.get_logger().error(
            'MANUAL_ESTOP — 자동복구 금지. 원인 확인 후 재개 버튼(DI16) 입력 대기 중.'
        )

    def manual_resume_callback(self, msg):
        if not self.manual_estop_pending:
            self.get_logger().info('재개 버튼 입력됨 — 대기 중인 수동 E-STOP이 없어 무시.')
            return

        # HOME으로 강제 복귀시키지 않는다 — 어느 위치로 갈지는 실행 스크립트(wok_test4)의
        # 단계 재시작이 결정한다. 여기서 HOME 이동을 하면 wok_test4의 재시작 동작과
        # 로봇을 동시에 제어하는 충돌이 난다.
        self.get_logger().info('재개 버튼 확인 — E-STOP 해제. 위치/단계는 실행 스크립트가 재시작.')
        self.reset_estop_pub.publish(Empty())
        self.calibrate_robot()
        self.manual_estop_pending = False
        self.get_logger().info('수동 재개 완료 — 작업을 재개합니다.')

    def reset_robot_to_home(self):
        self.get_logger().info('Resetting robot to HOME position.')
        reset_msg = String()
        reset_msg.data = 'HOME'
        self.reset_robot_pub.publish(reset_msg)
        time.sleep(1.0) # Simulate time to reach home

    def clear_safe_stop_only(self):
        # 홈 복귀 없이 로봇의 안전정지만 해제한다 (robot_command_bridge의 'CLEAR').
        # 이후 어디로 갈지는 wok_test4의 단계 재시작이 결정한다.
        self.get_logger().info('Clearing safe-stop only (no HOME move).')
        reset_msg = String()
        reset_msg.data = 'CLEAR'
        self.reset_robot_pub.publish(reset_msg)
        time.sleep(1.0)  # 안전정지 해제 완료까지 잠깐 대기

    def calibrate_robot(self):
        self.get_logger().info('Performing calibration.')
        self.calibrate_pub.publish(Empty())
        time.sleep(1.0) # Simulate calibration time

    def publish_auto_recovery_failed(self):
        alarm_msg = String()
        alarm_msg.data = 'AUTO_RECOVERY_FAILED'
        self.alarm_pub.publish(alarm_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RecoveryManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
