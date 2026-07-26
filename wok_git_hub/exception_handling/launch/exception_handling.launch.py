"""wok_exception_handling 노드 4개를 한 번에 실행하는 launch.

  ros2 launch wok_exception_handling exception_handling.launch.py

노드 7개:
  safety_monitor         — 감지 (gripper_state/force_feedback → alarm/estop)
  recovery_manager       — 복구 (alarm → 자동복구 or 관리자 대기)
  estop_button_io_integrate — 물리 버튼(DI13 정지/DI16 재개) + RG2 그리퍼(DI14 닫기/DI15 열기),
                              estop_button_io.py(move_stop 즉시정지)와
                              estop_button_io(1).py(RG2 제어)를 병합한 버전. 실로봇 필요
  shake_test_node        — 파지 안정성 검증(Shake Test), 실로봇 필요
  force_feedback_bridge  — 실로봇 힘/토크(get_tool_force)를 /force_feedback으로 발행, 실로봇 필요
  robot_command_bridge   — /reset_robot(HOME)을 받아 실제 안전정지 해제+movej 실행, 실로봇 필요
  robot_state_watchdog   — 로봇 자체 STATE_SAFE_STOP 전이를 직접 감지해 alarm/estop 발행, 실로봇 필요

DSR_ROBOT2를 쓰는 5개(estop_button_io_integrate/shake_test_node/force_feedback_bridge/
robot_command_bridge/robot_state_watchdog)는 로봇 bringup(dsr_bringup2)이
먼저 떠 있어야 정상 동작한다.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    common = dict(package='wok_exception_handling', output='screen')
    return LaunchDescription([
        Node(executable='safety_monitor', name='safety_monitor', **common),
        Node(executable='recovery_manager', name='recovery_manager', **common),
        Node(executable='shake_test_node', name='shake_test_node', **common),
        Node(executable='robot_command_bridge', name='robot_command_bridge', **common),

        # estop_button_io_integrate: 물리 버튼(DI13/16) 감시 + DI13 즉시 move_stop 호출
        #   + RG2 그리퍼(DI14 닫기/DI15 열기). 반드시 '별도 프로세스'로 떠야 한다 — 예전에
        #   이 로직을 wok_test4 안 백그라운드 스레드로 합쳤더니, rclpy가 같은 프로세스 안
        #   여러 스레드의 동시 spin(서로 다른 executor라도 컨텍스트 공유)을 안전하게
        #   지원하지 않아 wok_test4 자체가 시작하자마자 `IndexError: wait set index too big`로
        #   죽는 버그가 있었다. 별도 프로세스면 dsr_controller2가 공유 서비스라서 wok_test4가
        #   movel/movej로 블로킹돼 있어도 이 노드가 독립적으로 move_stop을 호출해 진짜
        #   즉시정지가 된다. 노드 이름은 기존 estop_button_io.py와 동일하게 유지해
        #   (executable만 교체) 그래프상 참조가 안 바뀌게 했다.
        Node(executable='estop_button_io_integrate', name='estop_button_io', **common),

        # ── dsr_controller2 서비스 폴링 부하를 줄이려고 잠시 꺼둔 노드들 ──
        #
        # force_feedback_bridge: get_tool_force 10Hz. FLOW_ANOMALY(soft)/shake_test 전용인데
        #   지금은 그 시나리오를 안 쓰므로 끔. FLOW_ANOMALY/shake 검증할 때 다시 켠다.
        # Node(executable='force_feedback_bridge', name='force_feedback_bridge', **common),
        #
        # robot_state_watchdog: get_robot_state 5Hz로 안전정지 감지 → alarm/estop을 발행해
        #   recovery_manager의 HUMAN_CONTACT 자동복구(안전정지 해제+calibrate)를 곧바로 태운다.
        #   wok_test4가 자기 체크포인트(매 movej/movel 직후)에서도 상태를 직접 확인해 스스로
        #   복구하므로, 이 노드까지 같이 켜면 두 복구 경로가 동시에 같은 로봇을 만지며 겹칠 수
        #   있다(레이스). wok_test4와 함께 쓸 땐 꺼두는 게 안전 — wait() 구간 중 걸린 안전정지는
        #   최대 그 wait만큼(수 초) 늦게 다음 체크포인트에서 잡히지만, 상태 자체는 계속
        #   안전정지로 남아있으므로 놓치지는 않는다.
        #   (wok_test4 없이 예외처리만 단독으로 돌릴 땐 다시 켜야 안전정지 감지가 된다.)
        # Node(executable='robot_state_watchdog', name='robot_state_watchdog', **common),
    ])
