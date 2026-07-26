"""wok_test4 — m0609_test4.drl 을 ROS2(DSR_ROBOT2)에서 실행.

프로세스: 재료 3종 투입(ingredients1~3) → 레버 열기 → 웍 파지 → 웍질(3세트 x 5회)
         → 웍 원위치 → 레버 닫기 → 홈.

원본 .drl 과 서브루틴/좌표/순서는 동일하며, ROS2 파이썬 API 비호환 인자만 제거:
  - movel 의 app_type=DR_MV_APP_NONE  → 제거
  - set_velx 의 세 번째 인자 DR_OFF    → 제거
  - movej 의 velx=1000.41 (wokking 내부, ROS2 API 미지원 인자) → 제거

안전: 첫 실행은 SPEED_RATIO 를 낮게(0.3) 두고 그리퍼 매핑·경로 확인 후 1.0 으로.
  ※ 실행 전: 작업 반경 비우기, 비상정지에 손, 그리퍼/레버 DO 매핑 사전 확인.
"""
import time

import rclpy
from rclpy.logging import get_logger
from std_msgs.msg import String, Bool, Empty, Float32
from geometry_msgs.msg import Point
from dsr_msgs2.srv import GetRobotState, SetRobotControl
from rokey.onrobot_rg2 import RG2Client, RG2Error, UnitId

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# ▼▼▼ 안전 파라미터 ▼▼▼
SPEED_RATIO = 0.3     # 전체 속도/가속 스케일. "원속도 그대로" 하려면 1.0
WOK_OUTER = 3          # 웍질 세트 수 (원본과 동일 3). 첫 테스트는 1 로 줄여도 됨
WOK_INNER = 5          # 세트당 웍질 반복 (원본과 동일 5)

# 그리퍼(OnRobot RG2) Compute Box — Modbus TCP로 폭/파지감지를 직접 읽는다.
# ★ 예전엔 /joint_states(finger_joint) → /onrobot/pose 서비스로 폭을 구했는데, 그 서비스가
#   응답하는 노드 없이 죽어있어(등록만 남고 호출하면 타임아웃) read_gripper_width_mm()이
#   항상 실패해 "낙관적으로 GRIPPED 처리" 폴백만 타고 있었다 — 즉 GRIP_FAILED 감지가 실질적으로
#   한 번도 동작한 적이 없었다. Compute Box에 직접 Modbus로 붙어서 이 문제를 근본적으로 없앤다.
# ★ Compute Box는 Modbus TCP 동시 연결을 1개만 허용한다(onrobot_rg2 client 문서 명시).
#   그래서 이 스크립트가 파지 확인 목적으로 자체 연결을 물고 있는 동안에는, onrobot_rg_control
#   ROS 드라이버(/onrobot/pose 등)를 동시에 띄우면 안 된다 — 연결이 하나만 허용되므로 충돌한다.
GRIPPER_MODBUS_HOST = "192.168.1.1"
GRIPPER_MODBUS_UNIT_ID = UnitId.QUICK_CHANGER   # 단일 Quick Changer 구성 (65)
# ▲▲▲

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
logger = get_logger("wok_test4")


class RestartPreparation(Exception):
    """준비 단계(홈 복귀~재료투입~레버열기~파지) 도중 안전정지가 걸리면
    이 예외를 던져서 그 단계를 처음부터(재료 투입 1/3부터) 다시 시작한다.
    중간부터 이어가면 재료가 실제로 제대로 들어갔는지 검증할 방법이 없기 때문."""
    pass


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("wok_test4", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    # --- wok_exception_handling 연동: 그리퍼 상태 발행 + E-STOP 정지/재개 ---
    gripper_state_pub = node.create_publisher(String, '/gripper_state', 10)
    alarm_pub = node.create_publisher(String, '/alarm', 10)
    estop_pub = node.create_publisher(Bool, '/estop', 10)
    reset_estop_pub = node.create_publisher(Empty, '/reset_estop', 10)
    _estop = {'active': False}
    _in_preparation = {'active': False}   # 재료투입~파지 준비단계 동안만 True

    # --- 관리자 대시보드 실시간 연동: 현재 단계 + TCP 위치 + 그리퍼 폭 발행 ---
    # backend/app.py의 ros_spin_thread가 이 토픽들을 구독해 robot_status에 반영하고,
    # /api/robot_status로 admin.html에 실시간 전달한다.
    stage_pub = node.create_publisher(String, '/wok_stage', 10)
    tcp_pub = node.create_publisher(Point, '/tcp_pose', 10)
    width_pub = node.create_publisher(Float32, '/gripper_width', 10)

    def publish_stage(name):
        logger.info(name)
        stage_pub.publish(String(data=name))

    def on_estop(msg):
        _estop['active'] = msg.data
        if msg.data:
            logger.warn('/estop 수신 — 다음 체크포인트에서 정지 대기')

    node.create_subscription(Bool, '/estop', on_estop, 10)

    # --- 그리퍼 폭/파지감지 읽기 (Modbus TCP 직결, 파지 성공/실패 판정용) ---
    # Compute Box와 소켓 연결을 한 번만 맺고 스크립트 끝까지 재사용한다(동시 연결 1개 제한).
    gripper_client = RG2Client(
        host=GRIPPER_MODBUS_HOST, unit_id=GRIPPER_MODBUS_UNIT_ID, timeout=1.0
    )
    try:
        gripper_client.connect()
        logger.info(f'그리퍼 Modbus 연결 성공 ({GRIPPER_MODBUS_HOST})')
    except RG2Error as e:
        logger.error(f'그리퍼 Modbus 연결 실패: {e} — 폭/파지감지 없이 진행합니다')

    def read_gripper_width_mm():
        """현재 그리퍼 폭(mm) 반환. 조회 실패 시 None."""
        if not gripper_client.connected:
            return None
        try:
            return gripper_client.get_width_mm(include_fingertip_offset=True)
        except RG2Error as e:
            logger.warn(f'그리퍼 폭 읽기 실패: {e}')
            return None

    def read_gripper_grip_detected():
        """그리퍼 하드웨어의 파지감지(grip_detected) 플래그. 조회 실패 시 None."""
        if not gripper_client.connected:
            return None
        try:
            return gripper_client.get_status().grip_detected
        except RG2Error as e:
            logger.warn(f'그리퍼 상태 읽기 실패: {e}')
            return None

    def publish_telemetry():
        """관리자 대시보드용 실시간 텔레메트리: 현재 TCP 위치 + 그리퍼 폭 발행.
        movel/movej 체크포인트(check_estop_and_wait)에서마다 호출된다 — 별도 스레드/루프
        없이 기존 체크포인트에 얹으므로 안전하다(추가 백그라운드 spin이 크래시를 유발했던
        전례가 있어 일부러 새 루프를 만들지 않았다)."""
        try:
            pos, _sol = get_current_posx()
            if pos is not None:
                tcp_pub.publish(Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
        except Exception:
            pass
        width = read_gripper_width_mm()
        if width is not None:
            width_pub.publish(Float32(data=float(width)))

    def publish_grip_result():
        """파지 명령 후 grip_detected(하드웨어 파지감지)로 GRIPPED/GRIP_FAILED를 판정해 발행한다.
        폭(width) 임계값 추정과 달리 grip_detected는 그리퍼가 실제로 뭔가에 걸려 멈췄는지를
        보여주는 하드웨어 신호라 손잡이 두께 등을 몰라도 정확하다.
        GRIP_FAILED면 safety_monitor→recovery_manager가 반응하도록 잠깐 기다린 뒤
        check_estop_and_wait()로 이 단계를 재시작시킨다."""
        width = read_gripper_width_mm()
        grip_detected = read_gripper_grip_detected()

        if grip_detected is None:
            logger.warn('그리퍼 상태 읽기 실패(Modbus 연결 확인) — 낙관적으로 GRIPPED 처리')
            gripper_state_pub.publish(String(data='GRIPPED'))
            return

        width_str = f'{width:.1f}mm' if width is not None else '읽기실패'
        if not grip_detected:
            logger.warn(f'파지 실패 감지 — grip_detected=False (폭={width_str}, 팬 못 잡음)')
            gripper_state_pub.publish(String(data='GRIP_FAILED'))
            # safety_monitor가 /estop을 세울 때까지 잠깐 대기 → 복구 사이클 후 이 단계 재시작
            deadline = time.time() + 2.0
            while time.time() < deadline and not _estop['active'] and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
            check_estop_and_wait()
        else:
            logger.info(f'파지 성공 — grip_detected=True (폭={width_str})')
            gripper_state_pub.publish(String(data='GRIPPED'))

    # 로봇 자체 안전정지(Safety Stop) 상태를 "직접" 조회한다.
    # 워치독 노드가 /estop을 발행해 주길 기다리는 대신, 매 체크포인트마다 로봇 상태를
    # 동기적으로 물어봐서 토픽 전달 타이밍에 의존하지 않고 확실히 잡는다.
    # (raw 서비스 + 타임아웃 → DSR_ROBOT2.get_robot_state()의 무한 대기 위험 회피)
    STATE_TRIGGER = {3, 5, 6, 9, 10}  # SAFE_OFF/SAFE_STOP/EMERGENCY_STOP/SAFE_STOP2/SAFE_OFF2
    # 아직 '동작 불가'라 해제 시퀀스가 더 필요한 상태들 (RECOVERY(8) 중간상태 포함).
    # 이 밖(STANDBY=1/MOVING=2/HOMMING=7 등)이면 동작 가능한 것으로 본다.
    STATE_NEEDS_CLEAR = {3, 5, 6, 8, 9, 10}
    # 상태별로 보내야 하는 SetRobotControl 값 (dsr_msgs2/SetRobotControl 정의 기준):
    #   3  SAFE_OFF    → 3(RESET_SAFET_OFF)    → STANDBY
    #   5  SAFE_STOP   → 2(RESET_SAFET_STOP)   → STANDBY
    #   9  SAFE_STOP2  → 4(RECOVERY_SAFE_STOP) → RECOVERY(8)
    #   10 SAFE_OFF2   → 5(RECOVERY_SAFE_OFF)  → RECOVERY(8)
    #   8  RECOVERY    → 7(RESET_RECOVERY)     → STANDBY
    #   6  EMERGENCY_STOP → 물리 비상정지 버튼을 풀어야만 해제됨 (소프트웨어 불가)
    STATE_TO_CONTROL = {3: 3, 5: 2, 9: 4, 10: 5, 8: 7}
    get_state_cli = node.create_client(GetRobotState, f'/{ROBOT_ID}/system/get_robot_state')
    set_ctrl_cli = node.create_client(SetRobotControl, f'/{ROBOT_ID}/system/set_robot_control')

    def read_robot_state():
        """로봇 현재 상태 코드를 반환. 조회 실패 시 None (오탐 방지)."""
        if not get_state_cli.service_is_ready():
            return None
        future = get_state_cli.call_async(GetRobotState.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=0.3)
        if not future.done():
            return None
        res = future.result()
        if res is None or not res.success:
            return None
        return res.robot_state

    # ★ 긴급정지(DI13/16) 및 (wait() 구간 중) 안전정지 블립 감지는 별도 프로세스인
    #   wok_exception_handling의 estop_button_io / robot_state_watchdog가 담당한다.
    #   이 안에 백그라운드 스레드로 넣었던 적이 있는데, rclpy가 같은 프로세스 안에서
    #   서로 다른 스레드가 동시에 spin(별도 executor라도 같은 컨텍스트 공유)하는 걸 안전하게
    #   지원하지 않아서 `IndexError: wait set index too big`로 이 스크립트 자체가
    #   시작하자마자 죽는 버그가 있었다. 이 프로세스는 항상 '메인 스레드 하나'에서만
    #   spin한다 — read_robot_state()/clear_safe_stop_until_ready()는 각 체크포인트에서
    #   동기적으로 호출되며, 그 사이 상태 변화는 별도 프로세스가 /estop 토픽으로 알려준다.

    def clear_safe_stop_until_ready():
        """로봇을 실제로 '동작 가능' 상태(STANDBY 등)로 되돌릴 때까지 반복 시도.

        ★ 상태마다 필요한 해제 명령이 다르다:
          SAFE_STOP(5)→2, SAFE_OFF(3)→3, SAFE_STOP2(9)→4, SAFE_OFF2(10)→5,
          그 뒤 RECOVERY(8)→7. EMERGENCY_STOP(6)은 물리 비상정지 버튼을 풀어야만 해제됨.
          (세게 치면 SAFE_STOP이 아니라 SAFE_OFF로 빠질 수 있어서 2만 보내면 안 풀린다.)

        ★ 왜 여기서(재시작 직전) 해제하나:
          해제해도 로봇을 트립된 자리에 가만히 두면 다시 안전정지로 돌아가는 경우가 있다.
          그래서 해제 '직후 곧바로' 다음 동작이 나가야 그 자리를 벗어나 해제가 유지된다 —
          여기서 해제하고 나면 호출부가 즉시 그 단계 첫 동작을 실행하므로 간격이 거의 0이다."""
        attempt = 0
        while rclpy.ok():
            attempt += 1
            state = read_robot_state()
            if state is not None and state not in STATE_NEEDS_CLEAR:
                logger.info(f'해제 확인 (robot_state={state}) — 즉시 재시작')
                return True

            control = STATE_TO_CONTROL.get(state)
            if control is not None and set_ctrl_cli.service_is_ready():
                req = SetRobotControl.Request()
                req.robot_control = control
                future = set_ctrl_cli.call_async(req)
                rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
            elif state == 6:  # EMERGENCY_STOP — 소프트웨어로 해제 불가
                logger.error('비상정지(EMERGENCY_STOP) 상태 — 물리 비상정지 버튼을 풀어주세요.')

            if attempt % 5 == 0:
                logger.warn(f'해제 재시도 {attempt}회 (robot_state={state}) — '
                            f'로봇이 물체에 눌려 있으면 물리적으로 치워주세요')
            if attempt >= 60:  # ~18초 이상 못 풀면 확인 없이 진행(무한대기 방지)
                logger.error('해제 확인 실패 — 확인 없이 재시작을 시도합니다')
                return False
            t = time.time() + 0.3
            while time.time() < t and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
        return False

    def check_estop_and_wait():
        """movel/movej 완료 직후마다 호출. E-STOP을 두 종류로 구분해 처리한다.

        A) 로봇이 실제 안전정지(SAFE_STOP/OFF 등)에 빠진 경우 — 물리 접촉/충돌이나
           DI13 긴급정지(estop_button_io가 move_stop으로 이미 즉시 정지시킴) 모두 결국
           로봇을 이 상태로 만든다: 3초 대기 → 직접 해제 → 그 단계 처음부터 재시작(자동복구).
        B) 로봇 상태는 멀쩡한데 /estop만 걸린 경우(예: recovery_manager가 아직 처리 중,
           또는 상태 전이가 늦게 반영됨): 외부에서 /estop 해제될 때까지 대기 후 재시작.
        DI13/DI16(물리 버튼)과 로봇 안전정지 블립 감지는 별도 프로세스
        (estop_button_io/robot_state_watchdog)가 담당하고 /estop 토픽으로 알려준다."""
        rclpy.spin_once(node, timeout_sec=0.0)
        publish_telemetry()  # 대시보드용 TCP 위치/그리퍼 폭 (매 동작 체크포인트마다)

        # 안전정지 감지 — 이 체크포인트에서 로봇 상태를 직접 조회
        state = read_robot_state()
        triggered = state in STATE_TRIGGER

        if triggered:
            # ── A) 안전정지 상태: 자동 해제 + 단계 재시작 ──
            if not _estop['active']:
                logger.error(f'로봇 안전정지 감지 (robot_state={state}) — 정지')
                _estop['active'] = True
                alarm_pub.publish(String(data='HUMAN_CONTACT'))  # 모니터링/로깅용 통지
                estop_pub.publish(Bool(data=True))

            logger.error('E-STOP(안전정지) — 3초 안정화 대기 후 해제 및 재시작.')
            end = time.time() + 3.0
            while time.time() < end and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

            clear_safe_stop_until_ready()       # 재시작 직전 해제 (해제 유지되게 곧바로 재개)
            _estop['active'] = False
            reset_estop_pub.publish(Empty())    # 예외처리 노드 동기화

            if _in_preparation['active']:
                raise RestartPreparation()

        elif _estop['active']:
            # ── B) 로봇은 정상인데 /estop만 걸려 있음: 외부에서 해제될 때까지 대기 ──
            # (MANUAL_ESTOP은 recovery_manager가 래치하고 DI16으로만 풀리며,
            #  그 해제는 /reset_estop → safety_monitor → /estop=False로 전달된다)
            logger.error('E-STOP 활성 — 자동복구 안 함. 외부 해제(재개)까지 대기합니다.')
            while _estop['active'] and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
            logger.info('E-STOP 해제(재개) 확인 — 현재 단계 재시작')
            if _in_preparation['active']:
                raise RestartPreparation()
    # ---------------------------------------------------------------

    from DSR_ROBOT2 import (
        set_singular_handling, set_velj, set_accj, set_velx, set_accx,
        movej as _movej, movel as _movel, move_periodic, mwait, wait, set_digital_output,
        posj, posx, get_current_posx,
        DR_AVOID, DR_MV_MOD_ABS, DR_MV_RA_DUPLICATE,
        ON, OFF,
    )

    WAIT_MOTION = 0.50   # 동작(movel/movej) 사이 wait_motion 시간. 웍질 반복 구간은 연속 동작 유지를 위해 제외.

    def movel(*args, **kwargs):
        _movel(*args, **kwargs)
        mwait(time=WAIT_MOTION)
        check_estop_and_wait()

    def movej(*args, **kwargs):
        _movej(*args, **kwargs)
        mwait(time=WAIT_MOTION)
        check_estop_and_wait()

    r = SPEED_RATIO

    def sc(x):
        """스칼라/리스트 모두 SPEED_RATIO 로 스케일."""
        return [v * r for v in x] if isinstance(x, (list, tuple)) else x * r

    logger.info("wok_test4 시작 — SPEED_RATIO=%.2f, WOK_OUTER=%d, WOK_INNER=%d"
                % (r, WOK_OUTER, WOK_INNER))

    set_singular_handling(DR_AVOID)
    set_velj(60.0 * r)
    set_accj(100.0 * r)
    set_velx(250.0 * r, 80.625 * r)
    set_accx(1000.0 * r, 322.5 * r)

    # ---- 서브루틴 ------------------------------------------------------

    def ingredients1():
        movel(posx(517.78, -645.92, 80.11, 0.86, 94.19, 88.94),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(615.78, -642.28, 73.26, 0.73, 94.19, 88.74),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(2, OFF)
        set_digital_output(3, ON)
        wait(3.00)
        movel(posx(632.87, -577.14, 373.27, 0.86, 93.62, 88.49),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(903.68, -48.07, 291.30, 31.02, 98.82, 93.12),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(903.68, -48.05, 291.33, 31.02, 98.82, -59.87),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        move_periodic(amp=[20.00, 0.00, 20.00, 0.00, 0.00, 0.00],
                       period=[0.50, 0.00, 0.50, 0.00, 0.00, 0.00],
                       atime=0.00, repeat=3, ref=0)
        mwait(time=0.50)
        movel(posx(904.95, -165.04, 270.56, 24.62, 98.82, 93.11),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(620.84, -582.39, 378.66, 2.92, 93.06, 88.11),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(615.78, -642.28, 73.25, 0.73, 94.19, 88.74),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)
        wait(3.00)
        movel(posx(517.78, -645.92, 80.11, 0.86, 94.19, 88.94),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def ingredients2():
        movel(posx(520.87, -533.60, 76.91, 1.00, 94.12, 89.16),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(620.73, -528.87, 70.86, 1.12, 94.10, 88.96),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(2, OFF)
        set_digital_output(3, ON)
        wait(3.00)
        movel(posx(620.73, -528.87, 370.86, 1.12, 94.10, 88.96),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(903.68, -48.07, 291.30, 31.02, 98.82, 93.12),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(903.68, -48.06, 291.30, 31.02, 98.82, -59.87),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        move_periodic(amp=[20.00, 0.00, 20.00, 0.00, 0.00, 0.00],
                       period=[0.50, 0.00, 0.50, 0.00, 0.00, 0.00],
                       atime=0.00, repeat=3, ref=0)
        mwait(time=0.50)
        movel(posx(904.95, -165.04, 270.56, 24.62, 98.82, 93.11),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(620.74, -528.86, 370.82, 1.12, 94.10, 88.96),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(623.80, -528.80, 72.11, 1.49, 93.96, 89.23),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)
        wait(3.00)
        movel(posx(532.01, -532.66, 79.35, 1.37, 94.09, 89.43),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def ingredients3():
        movel(posx(507.39, -433.03, 78.81, 1.59, 94.27, 89.62),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(614.61, -423.47, 73.75, 1.43, 94.14, 89.23),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(2, OFF)
        set_digital_output(3, ON)
        wait(3.00)
        movel(posx(614.61, -423.47, 370.75, 1.43, 94.14, 89.23),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(903.68, -48.07, 291.30, 31.02, 98.82, 93.12),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(903.68, -48.06, 291.30, 31.02, 98.82, -59.87),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        move_periodic(amp=[20.00, 0.00, 20.00, 0.00, 0.00, 0.00],
                       period=[0.50, 0.00, 0.50, 0.00, 0.00, 0.00],
                       atime=0.00, repeat=3, ref=0)
        mwait(time=0.50)
        movel(posx(904.95, -165.04, 270.56, 24.62, 98.82, 93.11),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(614.61, -423.47, 370.75, 1.43, 94.14, 89.23),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(614.61, -423.48, 73.78, 1.43, 94.14, 89.23),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)
        wait(3.00)
        movel(posx(507.40, -433.03, 78.81, 1.59, 94.27, 89.62),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def lever_open():
        movej(posj(-21.56, 45.45, 31.34, -0.05, 99.96, -16.28),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(posx(633.19, -239.32, 41.26, 163.12, -176.86, 168.11),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(3, OFF)
        set_digital_output(2, OFF)
        set_digital_output(1, ON)
        wait(3.00)
        movej(posj(-21.50, 45.35, 49.67, -0.04, 81.85, 69.22),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)
        wait(3.00)
        movej(posj(-21.56, 45.45, 31.34, -0.05, 99.96, -16.28),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def lever_close():
        movej(posj(-13.50, 0.37, 88.20, 24.61, 25.21, -28.38),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(posj(-21.56, 45.45, 31.34, -0.05, 99.96, -16.28),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(posx(633.19, -239.30, 41.23, 163.16, -176.86, -106.35),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(3, OFF)
        set_digital_output(2, OFF)
        set_digital_output(1, ON)
        wait(3.00)
        movej(posj(-21.50, 45.35, 49.67, -0.04, 81.85, -23.42),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)
        wait(3.00)
        movej(posj(-21.56, 45.45, 31.34, -0.05, 99.96, -16.28),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def wokking():
        for set_i in range(WOK_OUTER):
            for toss_i in range(WOK_INNER):
                publish_stage("웍질 세트 %d/%d, %d/%d" % (set_i + 1, WOK_OUTER, toss_i + 1, WOK_INNER))
                # 웍질 반복 구간: radius 블렌딩으로 이어지는 연속 동작이라 wait_motion 미적용 (raw _movel/_movej 사용)
                _movel(posx(698.57, -14.97, 330.02, 178.04, -117.09, 178.50),
                       vel=sc([1000.00, 120.00]), acc=sc([20000.00, 400.00]),
                       radius=30.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                _movej(posj(-1.83, 22.23, 81.03, -1.31, 13.97, -0.15),
                       vel=117.43 * r, acc=687.73 * r, radius=0, ra=DR_MV_RA_DUPLICATE)
                _movel(posx(846.38, -13.77, 336.99, 179.14, -84.31, 178.93),
                       vel=sc([2000.00, 225.00]), acc=sc([75000.00, 900.00]),
                       radius=30.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                _movel(posx(657.22, -18.37, 312.10, 177.96, -104.43, 178.63),
                       vel=sc([1000.00, 100.00]), acc=sc([20000.00, 400.00]),
                       radius=30.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                # 연속 블렌딩 동작 중간은 끊지 않고, 한 번의 웍질(toss) 완료 시점에만 /estop 체크
                check_estop_and_wait()

            movel(posx(702.86, -19.50, 180.97, 178.21, -110.23, 178.87),
                  radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
            wait(5.00)

        movel(posx(698.57, -14.97, 330.02, 178.04, -117.09, 178.50),
              vel=sc([1000.00, 120.00]), acc=sc([20000.00, 400.00]),
              radius=30.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movej(posj(11.32, -44.41, 118.37, 3.22, 36.75, -7.76),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(posx(413.20, 176.55, 392.31, 20.32, 99.59, -115.32),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movej(posj(11.56, -39.43, 124.18, 3.22, 25.94, -7.76),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(posj(-3.94, 3.44, 101.11, 3.22, 7.08, -7.76),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)

    # ---- 준비 단계 (홈 복귀~재료투입~레버열기~파지) ------------------------
    # 단계별 체크포인트: 안전정지가 걸리면 "이미 끝난 단계"는 다시 안 하고,
    # 걸린 시점의 그 단계만 처음부터 재시도한다 (전체를 1번으로 되돌리지 않음).

    def stage_home_and_ingredients1():
        movej(posj(0.00, 0.00, 90.00, 0.00, 90.00, 0.00), radius=0.00, ra=DR_MV_RA_DUPLICATE)
        publish_stage("재료 투입 1/3")
        ingredients1()

    def stage_ingredients2():
        publish_stage("재료 투입 2/3")
        ingredients2()

    def stage_ingredients3():
        publish_stage("재료 투입 3/3")
        ingredients3()
        mwait(time=1.00)

    def stage_lever_and_grip():
        publish_stage("레버 열기")
        lever_open()
        set_digital_output(2, OFF)
        wait(3.00)
        movej(posj(-4.10, -9.12, 133.07, -1.39, -17.77, -0.13),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        mwait(time=0.50)
        movel(posx(704.84, -21.97, 181.70, 178.01, -109.75, 178.85),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        mwait(time=0.50)
        set_digital_output(2, OFF)
        set_digital_output(1, ON)
        wait(3.00)
        # 그리퍼 폭(mm)을 읽어 파지 성공/실패를 판정한다 (GRIP_MIN_WIDTH_MM 미만이면 실패).
        # 실패 시 GRIP_FAILED를 발행해 safety_monitor→recovery_manager 복구 사이클을 태우고,
        # 복구(재파지) 후 이 단계(레버열기+파지)를 처음부터 재시작한다.
        publish_grip_result()

    PREP_STAGES = [
        ('홈복귀+재료투입1', stage_home_and_ingredients1),
        ('재료투입2', stage_ingredients2),
        ('재료투입3', stage_ingredients3),
        ('레버열기+파지', stage_lever_and_grip),
    ]

    def run_preparation():
        _in_preparation['active'] = True
        idx = 0
        while idx < len(PREP_STAGES):
            name, stage_fn = PREP_STAGES[idx]
            try:
                stage_fn()
                idx += 1
            except RestartPreparation:
                logger.warn(f'"{name}" 단계 중 안전정지 발생 — 이 단계부터 다시 시작합니다 '
                            f'(이전에 끝난 단계는 다시 안 함).')
                continue
        _in_preparation['active'] = False

    # ---- 메인 시퀀스 -----------------------------------------------------

    run_preparation()

    publish_stage("웍질 시작")
    wokking()

    movel(posx(704.48, -23.78, 192.98, 177.69, -112.93, 178.48),
          radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
    set_digital_output(1, OFF)
    set_digital_output(2, ON)
    wait(3.00)
    # 의도된 정상 릴리즈 — 'NOT_GRIPPED'/'GRIP_LOST'를 쓰면 safety_monitor가
    # "작업 중 파지 상실(PAN_DROPPED)"로 오인하므로 별도 상태값을 쓴다.
    gripper_state_pub.publish(String(data='RELEASED'))
    movel(posx(611.20, -15.91, 214.31, 178.09, -111.38, 178.50),
          radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    publish_stage("레버 닫기")
    lever_close()

    movej(posj(0.00, 0.00, 90.00, 0.00, 90.00, 0.00), radius=30.00, ra=DR_MV_RA_DUPLICATE)

    publish_stage("완료")
    gripper_client.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
