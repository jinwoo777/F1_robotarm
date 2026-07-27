"""wok_exception_handling 노드 4개를 한 번에 실행하는 launch.

  ros2 launch wok_exception_handling exception_handling.launch.py

현재 켜져 있는 노드 4개:
  safety_monitor         — 감지 (gripper_state/force_feedback → alarm/estop)
  recovery_manager       — 복구 (alarm → 자동복구 or 관리자 대기)
  robot_command_bridge   — /reset_robot(HOME)을 받아 실제 안전정지 해제+movej 실행, 실로봇 필요
  estop_button_io_integrate — 물리 버튼(DI13 정지/DI16 재개) + RG2 그리퍼(DI14 닫기/DI15 열기),
                              estop_button_io.py(move_stop 즉시정지)와
                              estop_button_io(1).py(RG2 제어)를 병합한 버전. 실로봇 필요

꺼둔 노드 3개 (이유는 아래 각 항목 주석 참고):
  shake_test_node        — ★ 웍 파손 원인. 재활성화 금지
  force_feedback_bridge  — 실로봇 힘/토크(get_tool_force)를 /force_feedback으로 발행
  robot_state_watchdog   — 로봇 자체 STATE_SAFE_STOP 전이를 직접 감지해 alarm/estop 발행

DSR_ROBOT2를 쓰는 노드(estop_button_io_integrate/robot_command_bridge, 꺼둔 것 중
shake_test_node/force_feedback_bridge/robot_state_watchdog)는 로봇 bringup(dsr_bringup2)이
먼저 떠 있어야 정상 동작한다.

■ 이 launch의 노드는 조리 노드(wok_integrate*)와 '같은 로봇'을 공유한다.
  모션을 직접 내보내는 노드를 켤 때는 조리 중 끼어들어도 안전한지 반드시 먼저 따질 것.
  shake_test_node가 정확히 이것 때문에 웍을 깨뜨렸다.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    common = dict(package='wok_exception_handling', output='screen')
    return LaunchDescription([
        Node(executable='safety_monitor', name='safety_monitor', **common),
        Node(executable='recovery_manager', name='recovery_manager', **common),
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

        # ── 끄기 전에 반드시 읽을 것 ────────────────────────────────────────
        #
        # shake_test_node: ★ 2026-07-27 실기에서 웍을 깨뜨린 원인이라 껐다. 재활성화 금지.
        #   /gripper_state == 'GRIPPED' 를 보면 곧바로 '별도 스레드에서' 같은 로봇에
        #   모션을 보낸다(movel 툴Z +20mm → move_periodic rx 5도 2회). 문제가 셋이다:
        #   1) 조리 노드(wok_integrate*)와 동시에 같은 로봇을 제어한다. wok_pick()이
        #      GRIPPED 를 발행하는 시점이 곧 뒤집기 시작 직전이라, 뒤집기 모션 도중
        #      상대이동이 끼어든다 — 상대이동의 기준 자세가 예상과 완전히 달라진다.
        #   2) LIFT_MM=20 을 '툴 Z = 위쪽'이라고 가정하는데(노드 주석에도 '가정'이라고 적혀
        #      있는 placeholder), 웍 파지 자세는 rx≈178도라 툴 Z 가 베이스 기준 아래를
        #      향한다. 그래서 '리프트'가 실제로는 하강이 되어 웍을 화구에 찍었다.
        #   3) 판정에 쓰는 /force_feedback 은 force_feedback_bridge 가 꺼져 있어 아무도
        #      발행하지 않는다 → 샘플 0개 → 항상 '불안정'으로 판정 → GRIP_UNSTABLE →
        #      safety_monitor 알람 → estop. 즉 위험한 모션을 하고 나서 매번 실패한다.
        #   되살리려면 최소한 (a) 조리 노드와 상호배제, (b) 툴 Z 방향 실측 확인,
        #   (c) force_feedback_bridge 동시 기동이 전부 선행되어야 한다.
        # Node(executable='shake_test_node', name='shake_test_node', **common),
        #
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
