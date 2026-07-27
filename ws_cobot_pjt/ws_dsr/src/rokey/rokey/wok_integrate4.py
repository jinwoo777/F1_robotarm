"""wok_integrate4 — wok_integrate2에 probe_flip3 / wok_final3의 하드웨어 검증 반영분을 이식.

wok_integrate2 = probe_flip2(부침개) + wok_final(볶음밥) + wok_test4(예외처리) 통합본이었다.
이 파일은 그 뒤 실기에서 잡힌 문제들을 고친 probe_flip3 / wok_final3의 변경분만 반영한
버전이다 (안전 인프라·dish 분기 구조는 wok_integrate2 그대로).

■ wok_final3 → 반영 (볶음밥/공통 경로)
  1) 웍 내려놓기 경유점 P_WOK_PLACE_VIA(Z=286) 추가.
     웍을 내려놓을 때 직선으로 바로 내려가면 화로에 걸려서, 이 지점을 먼저 거친 뒤
     P_WOK_GRIP으로 내려간다. wok_place() 와 본 웍질 세트 휴지 양쪽에 적용.
  2) toss() 마지막 회차 radius=0.
     블렌딩 관성 없이 완전히 멈춘 뒤 movel(P_WOK_GRIP)이 진짜 직선으로 나가도록 —
     (1)과 함께 화로 간섭 방지.
  3) scoop() 주걱 파지 자세 변경 + 빼는 동작을 movel(P_SCOOP_TRANSIT_A) 대신 movej로.
     파지 자세: (-39.69, 32.44, 78.09, -0.24, 69.01, 53.35) → (-40.11, 35.35, 73.83, 0.66, 73.86, 51.94)
     빼는 위치: movej(-39.84, 31.64, 73.83, 0.36, 75.1, 51.94) 신규.
     P_SCOOP_TRANSIT_A는 이제 복귀 경로에서만 쓰인다.
  4) scoop() 힘제어 강화:
       - 목표력 -5N → -10N
       - check_force_condition 원샷 체크 → 0.5초 간격 20회(약 10초) 폴링 + get_tool_force 로깅.
         힘이 늦게 오르거나 순간적으로 스쳐 지나가면 원샷 체크는 놓치기 쉬움.
       - 판정 상한 max=8 → 15
       - 젓기를 move_spiral → move_periodic(ref=DR_TOOL)으로 교체.
         move_spiral은 이 코드베이스에서 실동작이 검증된 적이 없고(회전 평면이 팬 표면에
         막히면 움직임이 사라짐), move_periodic은 shake_ladle/gear_force에서 검증됨.
         → move_spiral import 제거, get_tool_force / DR_TOOL import 추가.
  5) 붓기 목표 좌표 (413.20, 176.55, 392.31, 20.32, 115.00, -115.32)
     → (393.35, 208.74, 398.03, 24.66, 101.59, -137.26).
     이로써 볶음밥 붓기와 부침개 붓기가 "붓고 나서 70mm 들어올리는지" 하나만 달라져,
     두 함수를 pour_to_bowl(lift_after)로 합쳤다.

■ probe_flip3 → 반영 (부침개 경로)
  6) FLIP_SPEED_RATIO 0.3 → 0.6 (재료 픽업 속도와 통일). probe_flip3의 유일한 변경점.

ROS 파라미터 `dish`로 조리를 선택한다 (기본값 fried_rice):
    ros2 run rokey wok_integrate4 --ros-args -p dish:=fried_rice
    ros2 run rokey wok_integrate4 --ros-args -p dish:=jeon

┌─ fried_rice (wok_final3 기준) ───────────────────────────────────────────┐
│ 재료 3종 투입 → 레버 열기 → 웍 파지 → 1차 웍질 → scoop(힘제어 국자삽입)   │
│ → 웍 재파지 → 본 웍질(WOK_OUTER×WOK_INNER) → 붓기 → 웍 원위치 → 레버 닫기 │
└───────────────────────────────────────────────────────────────────────────┘
┌─ jeon (probe_flip3 기준) ──────────────────────────────────────────────────┐
│ 재료 투입(반죽, ingredients2 재사용) → 레버 열기 → 웍 파지 → flip 반복     │
│ (FLIP_COUNT) → 그릇에 붓기 → 웍 원위치 → 레버 닫기                        │
└───────────────────────────────────────────────────────────────────────────┘

두 조리 모두 wok_test4의 안전 인프라를 동일하게 사용한다 (wok_integrate2와 동일):
  - check_estop_and_wait(): 모든 movel/movej 완료 직후 안전정지 감지 + 자동 해제/복구
  - RestartPreparation: 준비 단계 중 안전정지가 걸리면 그 단계만 재시작
  - 그리퍼 Modbus(RG2Client) grip_detected로 실제 파지 성공/실패 판정 + 웍 파지 폭 검증
  - /wok_stage, /tcp_pose, /gripper_width 퍼블리시 (backend 대시보드 연동)

주의: wok_final3/probe_flip3의 변경 중 붓기 좌표·경유점 Z·힘제어 파라미터는 실기에서
확인이 필요한 값이다. 처음 실행은 SPEED_RATIO/FLIP_SPEED_RATIO를 낮춰 경로부터 확인할 것.
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

# ▼▼▼ 볶음밥(fried_rice) 파라미터 — wok_final3.py 기준 ▼▼▼
SIMULATION = False     # True → scoop()의 힘제어 구간 우회 (시뮬레이터 검증용)
SPEED_RATIO = 0.6
INGREDIENT_SPEED_RATIO = 0.6
TOSS_SPEED_RATIO = 1.25
WOK_OUTER = 3
WOK_INNER = 7
GRIP_WAIT = 1.50
WOK_GRIP_WAIT = 1.50
SET_REST_WAIT = 5.00
# ▲▲▲

# ▼▼▼ scoop 힘제어 파라미터 — wok_final3에서 상향된 값 ▼▼▼
SCOOP_FORCE_N = -10.00        # 목표력 Z (wok_final: -5.00 → wok_final3: -10.00)
SCOOP_FC_MIN = 2              # check_force_condition 하한
SCOOP_FC_MAX = 15             # 상한 (wok_final: 8 → wok_final3: 15)
SCOOP_FC_POLL_COUNT = 20      # 0.5초 간격 20회 = 약 10초 폴링 (원샷 체크 대체)
SCOOP_FC_POLL_INTERVAL = 0.5
# ▲▲▲

# ▼▼▼ 웍(프라이팬) 파지 폭 검증 — 실제 파지 실패가 잦아 추가한 예외 처리 ▼▼▼
# 정상 파지 시 그리퍼 폭은 56mm 근방이어야 한다. 이 범위를 벗어나면 웍 테두리를 잘못
# 물었거나 헛잡은 것으로 보고, 열었다 닫았다를 재시도한다.
WOK_GRIP_WIDTH_TARGET_MM = 56.0
WOK_GRIP_WIDTH_TOLERANCE_MM = 2.0
WOK_GRIP_WIDTH_RETRY = 3

# 주걱(국자) 파지 폭 검증 — 웍과 같은 조건(56±2mm)으로 맞춤(요청 반영).
# 주걱 손잡이 두께가 웍 손잡이와 다르면 실측해서 여기만 따로 조정하면 된다.
# 재시도 횟수(WOK_GRIP_WIDTH_RETRY)와 실패 시 처리(비상정지)는 웍과 공용이다.
SCOOP_GRIP_WIDTH_TARGET_MM = 56.0
SCOOP_GRIP_WIDTH_TOLERANCE_MM = 2.0
# ▲▲▲

# ▼▼▼ 부침개(jeon) 파라미터 — probe_flip3.py 기준 ▼▼▼
FLIP_SPEED_RATIO = 0.6          # 홈/레버 이동 속도. probe_flip3에서 0.3 → 0.6 (재료 픽업 속도와 통일)
FLIP_COUNT = 3
FLIP_SET_REST_WAIT = 3.00       # flip 사이 대기. 원본엔 FLIP_PAUSE_SEC가 선언만 되고 실제로는
                                # 안 쓰이는 죽은 상수였음 — 실제 사용값(3.00)으로 통일해 이름 대체.
VELOCITY_J = 10                 # ready_joint 진입 전용 (느려도 됨)
ACC_J = 20
WOK_VEL_J = 500                 # flap/flat(던지고 받기) 전용
WOK_ACC_J = 1000
READY_JOINT = [3.01, 8.44, 53.86, -2.55, 72.76, 3.06]
FLAP_JOINT = [3.01, 5.0, 43.00, -2.55, 45, 3.06]
FLAT_JOINT = [3.01, 5.0, 43.00, -2.55, 50, 3.06]
# ▲▲▲

WAIT_MOTION = 0.50   # movel/movej 사이 공통 wait_motion 시간. 연속 블렌딩 구간(toss)만 제외.

# 그리퍼(OnRobot RG2) Compute Box — Modbus TCP 직결 (wok_test4.py와 동일 이유:
# /onrobot/pose 서비스가 죽어있어 폭/파지감지를 못 읽던 문제를 근본적으로 회피).
# Compute Box는 동시 연결 1개만 허용하므로 onrobot_rg_control 드라이버와 동시 구동 금지.
GRIPPER_MODBUS_HOST = "192.168.1.1"
GRIPPER_MODBUS_UNIT_ID = UnitId.QUICK_CHANGER

# ─────────────────────────────────────────────────────────────────────────
# 반복 좌표 상수 (wok_final3.py 기준). main() 안에서 posj()/posx()로 래핑된다.
# ─────────────────────────────────────────────────────────────────────────
_J_HOME             = [0.00, 0.00, 90.00, 0.00, 90.00, 0.00]
_J_LEVER_NEUTRAL    = [-21.56, 45.45, 31.34, -0.05, 99.96, -16.28]
_J_LEVER_OPEN_NEUT  = [-20.55, 43.31, 29.64, -0.39, 106.07, -23.54]
_J_WOK_APPROACH     = [-4.10, -9.12, 133.07, -1.39, -17.77, -0.13]   # = probe_flip 접근 관절
_J_TOSS_A           = [-3.40, 8.62, 81.29, -0.28, 42.61, -0.05]
_J_TOSS_B           = [-3.34, -3.85, 81.09, -0.24, 55.05, -0.05]
_J_SCOOP_STANDBY    = [-39.77, 28.57, 55.08, -0.24, 96.06, 53.34]
_J_SCOOP_TRANSIT    = [-47.32, 24.05, 36.21, -0.33, 119.71, 45.99]
_J_SCOOP_APPROACH   = [-4.11, 23.42, 36.90, 8.99, 67.66, 74.51]
# wok_final3에서 변경된 주걱 파지 자세 (원본 -39.69, 32.44, 78.09, -0.24, 69.01, 53.35)
_J_SCOOP_GRIP       = [-40.11, 35.35, 73.83, 0.66, 73.86, 51.94]
# 주걱을 잡고 빼는 위치. wok_final3에서 movel(P_SCOOP_TRANSIT_A)를 이 movej로 대체함.
_J_SCOOP_PULLOUT    = [-39.84, 31.64, 73.83, 0.36, 75.10, 51.94]

# ★ 웍 파지 = 웍 내려놓기 = probe_flip의 stove_pos, 전부 이 좌표로 통일.
_P_WOK_GRIP         = [704.84, -21.97, 181.70, 178.01, -109.75, 178.85]
# P_WOK_GRIP보다 높은(Z 286) 경유점 (wok_final3 신규). 웍을 내려놓을 때 직선으로 바로
# 내려가면 화로에 걸려서, 이 지점을 먼저 거친 뒤 P_WOK_GRIP으로 내려간다.
_P_WOK_PLACE_VIA    = [703.08, -21.33, 286.00, 177.81, -114.16, 179.21]
# 본 웍질 후 웍을 내려놓고 레버로 이동하기 전 경유점 (wok_test4/wok_final/probe_flip 공통).
_P_POST_RELEASE     = [611.20, -15.91, 214.31, 178.09, -111.38, 178.50]

_P_POUR_APPROACH    = [903.68, -48.07, 291.30, 31.02, 98.82, 93.12]
_P_POUR_ROT         = [903.68, -48.06, 291.30, 31.02, 98.82, -59.87]
_P_POUR_EXIT        = [904.95, -165.04, 270.56, 24.62, 98.82, 93.11]

_P_ING1_STANDBY     = [517.78, -645.92, 80.11, 0.86, 94.19, 88.94]
_P_ING1_GRIP        = [615.78, -642.28, 73.26, 0.73, 94.19, 88.74]
_P_ING2_LIFT        = [620.73, -528.87, 370.86, 1.12, 94.10, 88.96]
_P_ING3_STANDBY     = [507.39, -433.03, 78.81, 1.59, 94.27, 89.62]
_P_ING3_GRIP        = [614.61, -423.47, 73.75, 1.43, 94.14, 89.23]
_P_ING3_LIFT        = [614.61, -423.47, 370.75, 1.43, 94.14, 89.23]

_P_SCOOP_TRANSIT_A  = [439.91, -356.18, 26.75, 136.39, -179.80, -130.78]  # (1회, 복귀 경로)
_P_SCOOP_TRANSIT_B  = [335.25, -356.98, 32.47, 168.59, 179.74, -98.19]    # (2회)

# 붓기 (wok_final3에서 볶음밥/부침개 목표 좌표가 같아짐 — pour_to_bowl()에서 공용)
_P_POUR_LIFTOFF     = [698.57, -14.97, 330.02, 178.04, -117.09, 178.50]   # 웍을 들고 화로에서 빠져나옴
_P_POUR_BOWL        = [393.35, 208.74, 398.03, 24.66, 101.59, -137.26]    # 그릇 위 붓기 자세
_J_POUR_APPROACH    = [11.32, -44.41, 118.37, 3.22, 36.75, -7.76]
_J_POUR_RETRACT     = [11.56, -39.43, 124.18, 3.22, 25.94, -7.76]
_J_POUR_RETURN      = [-3.94, 3.44, 101.11, 3.22, 7.08, -7.76]

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
logger = get_logger("wok_integrate4")


class RestartPreparation(Exception):
    """준비 단계 도중 안전정지가 걸리면 이 예외를 던져 그 단계만 처음부터 다시 시작한다.
    중간부터 이어가면 재료/파지가 실제로 제대로 됐는지 검증할 방법이 없기 때문(wok_test4와 동일)."""
    pass


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("wok_integrate4", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    node.declare_parameter('dish', 'fried_rice')
    dish = node.get_parameter('dish').value
    if dish not in ('fried_rice', 'jeon'):
        logger.error(f"알 수 없는 dish 파라미터 '{dish}' — fried_rice로 진행합니다")
        dish = 'fried_rice'

    # ── wok_exception_handling 연동: 그리퍼 상태 발행 + E-STOP 정지/재개 ──
    gripper_state_pub = node.create_publisher(String, '/gripper_state', 10)
    alarm_pub = node.create_publisher(String, '/alarm', 10)
    estop_pub = node.create_publisher(Bool, '/estop', 10)
    reset_estop_pub = node.create_publisher(Empty, '/reset_estop', 10)
    _estop = {'active': False}
    _in_preparation = {'active': False}

    # ── 관리자 대시보드 실시간 연동 (topic 이름은 wok_test4와 동일하게 유지) ──
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

    # ── 그리퍼 폭/파지감지 (Modbus TCP 직결, 파지 성공/실패 판정용) ──
    gripper_client = RG2Client(
        host=GRIPPER_MODBUS_HOST, unit_id=GRIPPER_MODBUS_UNIT_ID, timeout=1.0
    )
    try:
        gripper_client.connect()
        logger.info(f'그리퍼 Modbus 연결 성공 ({GRIPPER_MODBUS_HOST})')
    except RG2Error as e:
        logger.error(f'그리퍼 Modbus 연결 실패: {e} — 폭/파지감지 없이 진행합니다')

    def read_gripper_width_mm():
        if not gripper_client.connected:
            return None
        try:
            return gripper_client.get_width_mm(include_fingertip_offset=True)
        except RG2Error as e:
            logger.warn(f'그리퍼 폭 읽기 실패: {e}')
            return None

    def read_gripper_grip_detected():
        if not gripper_client.connected:
            return None
        try:
            return gripper_client.get_status().grip_detected
        except RG2Error as e:
            logger.warn(f'그리퍼 상태 읽기 실패: {e}')
            return None

    def publish_telemetry():
        """대시보드용 TCP 위치/그리퍼 폭. movel/movej 체크포인트에만 얹는다 — 별도
        백그라운드 스레드는 rclpy의 동시 spin 문제로 이 프로세스를 죽인 전례가 있어 금지."""
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
        """grip_detected(하드웨어 파지감지)로 GRIPPED/GRIP_FAILED를 판정해 발행한다.
        폭 임계값 추정과 달리 그리퍼가 실제로 뭔가에 걸려 멈췄는지를 보는 하드웨어 신호라
        정확하다. GRIP_FAILED면 safety_monitor→recovery_manager가 반응할 시간을 준 뒤
        check_estop_and_wait()로 재시작 여부를 맡긴다."""
        width = read_gripper_width_mm()
        grip_detected = read_gripper_grip_detected()

        if grip_detected is None:
            logger.warn('그리퍼 상태 읽기 실패(Modbus 연결 확인) — 낙관적으로 GRIPPED 처리')
            gripper_state_pub.publish(String(data='GRIPPED'))
            return

        width_str = f'{width:.1f}mm' if width is not None else '읽기실패'
        if not grip_detected:
            logger.warn(f'파지 실패 감지 — grip_detected=False (폭={width_str})')
            gripper_state_pub.publish(String(data='GRIP_FAILED'))
            deadline = time.time() + 2.0
            while time.time() < deadline and not _estop['active'] and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
            check_estop_and_wait()
        else:
            logger.info(f'파지 성공 — grip_detected=True (폭={width_str})')
            gripper_state_pub.publish(String(data='GRIPPED'))

    # ── 로봇 안전정지 상태를 매 체크포인트마다 직접 폴링 (토픽 전달 타이밍에 의존 안 함) ──
    STATE_TRIGGER = {3, 5, 6, 9, 10}       # SAFE_OFF/SAFE_STOP/EMERGENCY_STOP/SAFE_STOP2/SAFE_OFF2
    STATE_NEEDS_CLEAR = {3, 5, 6, 8, 9, 10}
    STATE_TO_CONTROL = {3: 3, 5: 2, 9: 4, 10: 5, 8: 7}
    get_state_cli = node.create_client(GetRobotState, f'/{ROBOT_ID}/system/get_robot_state')
    set_ctrl_cli = node.create_client(SetRobotControl, f'/{ROBOT_ID}/system/set_robot_control')

    def read_robot_state():
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

    def clear_safe_stop_until_ready():
        """'동작 가능' 상태로 되돌릴 때까지 반복 시도. 상태마다 필요한 해제 명령이 다르고
        (EMERGENCY_STOP은 물리 버튼 해제만 가능), 해제 직후 곧바로 다음 동작이 나가야
        그 자리를 벗어나 해제가 유지된다 — 호출부가 반환 즉시 다음 동작을 실행한다."""
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
            elif state == 6:
                logger.error('비상정지(EMERGENCY_STOP) 상태 — 물리 비상정지 버튼을 풀어주세요.')

            if attempt % 5 == 0:
                logger.warn(f'해제 재시도 {attempt}회 (robot_state={state})')
            if attempt >= 60:
                logger.error('해제 확인 실패 — 확인 없이 재시작을 시도합니다')
                return False
            t = time.time() + 0.3
            while time.time() < t and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
        return False

    def check_estop_and_wait():
        """movel/movej 완료 직후마다 호출. A) 로봇이 실제 안전정지 상태면 3초 대기→해제→
        준비 단계 중이면 그 단계 재시작. B) 로봇은 정상인데 /estop만 걸려 있으면 외부
        해제까지 대기 후 재시작. DI13/DI16 및 안전정지 블립 감지는 별도 프로세스
        (estop_button_io/robot_state_watchdog)가 담당하고 /estop 토픽으로 알려준다."""
        rclpy.spin_once(node, timeout_sec=0.0)
        publish_telemetry()

        state = read_robot_state()
        triggered = state in STATE_TRIGGER

        if triggered:
            if not _estop['active']:
                logger.error(f'로봇 안전정지 감지 (robot_state={state}) — 정지')
                _estop['active'] = True
                alarm_pub.publish(String(data='HUMAN_CONTACT'))
                estop_pub.publish(Bool(data=True))

            logger.error('E-STOP(안전정지) — 3초 안정화 대기 후 해제 및 재시작.')
            end = time.time() + 3.0
            while time.time() < end and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)

            clear_safe_stop_until_ready()
            _estop['active'] = False
            reset_estop_pub.publish(Empty())

            if _in_preparation['active']:
                raise RestartPreparation()

        elif _estop['active']:
            logger.error('E-STOP 활성 — 자동복구 안 함. 외부 해제(재개)까지 대기합니다.')
            while _estop['active'] and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
            logger.info('E-STOP 해제(재개) 확인 — 현재 단계 재시작')
            if _in_preparation['active']:
                raise RestartPreparation()

    def trigger_estop_and_wait(reason):
        """물리 비상정지 버튼을 누른 것과 동일하게 /estop을 발행하고, 외부(관리자)가 재개
        (estop 해제)할 때까지 자동복구 없이 대기한다. check_estop_and_wait()의 '이미 활성'
        분기와 동일한 패턴 — 로봇 자체는 정상이어도(예: 파지 폭 이상처럼 로봇 상태로는
        안 잡히는 문제) 사람이 직접 확인해야 하는 상황에 쓴다. 준비 단계 중이면 재개 후
        그 단계를 처음부터 다시 시작한다."""
        if not _estop['active']:
            logger.error(f'{reason} — 비상정지 발행')
            _estop['active'] = True
            alarm_pub.publish(String(data='GRIP_WIDTH_ABNORMAL'))
            estop_pub.publish(Bool(data=True))

        logger.error('E-STOP 활성 — 자동복구 안 함. 외부 해제(재개)까지 대기합니다.')
        while _estop['active'] and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
        logger.info('E-STOP 해제(재개) 확인 — 현재 단계 재시작')
        if _in_preparation['active']:
            raise RestartPreparation()

    def run_preparation(stages):
        """단계별 체크포인트: 안전정지가 걸리면 걸린 시점의 그 단계만 처음부터 재시도."""
        _in_preparation['active'] = True
        idx = 0
        while idx < len(stages):
            name, stage_fn = stages[idx]
            try:
                stage_fn()
                idx += 1
            except RestartPreparation:
                logger.warn(f'"{name}" 단계 중 안전정지 발생 — 이 단계부터 다시 시작합니다.')
                continue
        _in_preparation['active'] = False

    # ─────────────────────────────────────────────────────────────────────
    # wok_final3 기준 import: move_spiral 제거(젓기를 move_periodic으로 교체),
    # get_tool_force / DR_TOOL 추가(힘제어 폴링 로깅 + 툴 좌표계 젓기).
    from DSR_ROBOT2 import (
        set_singular_handling, set_velj, set_accj, set_velx, set_accx,
        movej as _movej, movel as _movel, move_periodic,
        mwait, wait, set_digital_output,
        task_compliance_ctrl, release_compliance_ctrl, set_stiffnessx,
        set_desired_force, release_force, check_force_condition, get_tool_force,
        posj, posx, get_current_posx,
        DR_AVOID, DR_MV_MOD_ABS, DR_MV_MOD_REL, DR_MV_RA_DUPLICATE, DR_TOOL,
        ON, OFF,
    )

    # 힘제어 상수는 설치 버전에 따라 없을 수 있어 방어적으로 처리 (wok_final3과 동일 패턴).
    try:
        from DSR_ROBOT2 import DR_AXIS_Z as _DR_AXIS_Z
    except ImportError:
        _DR_AXIS_Z = 2
    try:
        from DSR_ROBOT2 import DR_BASE as _DR_BASE
    except ImportError:
        _DR_BASE = 0
    try:
        from DSR_ROBOT2 import DR_FC_MOD_REL as _DR_FC_MOD_REL
    except ImportError:
        _DR_FC_MOD_REL = 1

    # DSR_ROBOT2는 import 시점에 movej/movel/set_singular_handling 등 모든 서비스
    # 클라이언트를 한꺼번에 생성한다. 서버(dsr_controller2)가 이미 몇 분째 떠 있어도
    # '이 클라이언트'는 방금 막 생성된 것이라 DDS 매칭이 아직 안 끝났을 수 있고, 매칭 전에
    # call_async()가 나가면 기본 QoS(RELIABLE+VOLATILE)라 그 요청은 재전송 없이 유실되어
    # future가 영원히 안 풀린다 — set_singular_handling(DR_AVOID)이 매번 같은 지점에서
    # 멈추던 원인이 이것이었다(wok_test4에서도 동일 현상 재현 확인). 첫 모션 호출 전에
    # 매칭이 끝나도록 짧게 대기한다.
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(1.0)

    # posj/posx 래핑 ---------------------------------------------------------
    J_HOME            = posj(_J_HOME)
    J_LEVER_NEUTRAL   = posj(_J_LEVER_NEUTRAL)
    J_LEVER_OPEN_NEUT = posj(_J_LEVER_OPEN_NEUT)
    J_WOK_APPROACH    = posj(_J_WOK_APPROACH)
    J_TOSS_A          = posj(_J_TOSS_A)
    J_TOSS_B          = posj(_J_TOSS_B)
    J_SCOOP_STANDBY   = posj(_J_SCOOP_STANDBY)
    J_SCOOP_TRANSIT   = posj(_J_SCOOP_TRANSIT)
    J_SCOOP_APPROACH  = posj(_J_SCOOP_APPROACH)
    J_SCOOP_GRIP      = posj(_J_SCOOP_GRIP)
    J_SCOOP_PULLOUT   = posj(_J_SCOOP_PULLOUT)
    J_POUR_APPROACH   = posj(_J_POUR_APPROACH)
    J_POUR_RETRACT    = posj(_J_POUR_RETRACT)
    J_POUR_RETURN     = posj(_J_POUR_RETURN)

    P_WOK_GRIP        = posx(_P_WOK_GRIP)
    P_WOK_PLACE_VIA   = posx(_P_WOK_PLACE_VIA)
    P_POST_RELEASE    = posx(_P_POST_RELEASE)
    P_POUR_APPROACH   = posx(_P_POUR_APPROACH)
    P_POUR_ROT        = posx(_P_POUR_ROT)
    P_POUR_EXIT       = posx(_P_POUR_EXIT)
    P_ING1_STANDBY    = posx(_P_ING1_STANDBY)
    P_ING1_GRIP       = posx(_P_ING1_GRIP)
    P_ING2_LIFT       = posx(_P_ING2_LIFT)
    P_ING3_STANDBY    = posx(_P_ING3_STANDBY)
    P_ING3_GRIP       = posx(_P_ING3_GRIP)
    P_ING3_LIFT       = posx(_P_ING3_LIFT)
    P_SCOOP_TRANSIT_A = posx(_P_SCOOP_TRANSIT_A)
    P_SCOOP_TRANSIT_B = posx(_P_SCOOP_TRANSIT_B)
    P_POUR_LIFTOFF    = posx(_P_POUR_LIFTOFF)
    P_POUR_BOWL       = posx(_P_POUR_BOWL)

    READY_J = posj(*READY_JOINT)
    FLAP_J = posj(*FLAP_JOINT)
    FLAT_J = posj(*FLAT_JOINT)

    # 공통 래퍼: 모든 movel/movej는 모션 완료 후 estop 체크포인트를 거친다
    # (toss()의 연속 블렌딩 구간만 raw _movel/_movej로 예외 — wok_test4/wok_final과 동일 이유).
    def movel(*a, **kw):
        _movel(*a, **kw)
        mwait(time=WAIT_MOTION)
        check_estop_and_wait()

    def movej(*a, **kw):
        _movej(*a, **kw)
        mwait(time=WAIT_MOTION)
        check_estop_and_wait()

    r = SPEED_RATIO
    tr = TOSS_SPEED_RATIO

    def sc_toss(x):
        return [v * tr for v in x] if isinstance(x, (list, tuple)) else x * tr

    def grip_close():
        """DO1=닫기 (레버/웍 파지)."""
        set_digital_output(2, OFF)
        set_digital_output(3, OFF)
        set_digital_output(1, ON)

    def grip_open():
        """DO2=열기."""
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)

    def grip_ladle():
        """DO3=국자/재료 파지."""
        set_digital_output(1, OFF)
        set_digital_output(2, OFF)
        set_digital_output(3, ON)

    def shake_ladle():
        ret = move_periodic(amp=[20.00, 0.00, 20.00, 0.00, 0.00, 0.00],
                             period=[0.50, 0.00, 0.50, 0.00, 0.00, 0.00],
                             atime=0.00, repeat=3, ref=0)
        if ret != 0:
            logger.warn("재료 투입 흔들기(move_periodic) 실패 — ret=%s" % ret)
        mwait(time=0.50)

    logger.info("wok_integrate4 시작 — dish=%s, SPEED_RATIO=%.2f, FLIP_SPEED_RATIO=%.2f, SIMULATION=%s"
                % (dish, r, FLIP_SPEED_RATIO, SIMULATION))

    set_singular_handling(DR_AVOID)

    # ── 볶음밥(fried_rice) 서브루틴 (wok_final3.py 기준) ───────────────────

    def ingredients1():
        movel(P_ING1_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING1_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_ladle()
        wait(GRIP_WAIT)
        movel(posx(632.87, -577.14, 373.27, 0.86, 93.62, 88.49),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_APPROACH, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_ROT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        shake_ladle()
        movel(P_POUR_EXIT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(620.84, -582.39, 378.66, 2.92, 93.06, 88.11),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING1_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        movel(P_ING1_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def ingredients2():
        movel(posx(520.87, -533.60, 76.91, 1.00, 94.12, 89.16),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(620.73, -528.87, 70.86, 1.12, 94.10, 88.96),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_ladle()
        wait(GRIP_WAIT)
        movel(P_ING2_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_APPROACH, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_ROT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        shake_ladle()
        movel(P_POUR_EXIT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING2_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(623.80, -528.80, 72.11, 1.49, 93.96, 89.23),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        movel(posx(532.01, -532.66, 79.35, 1.37, 94.09, 89.43),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def ingredients3():
        movel(P_ING3_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING3_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_ladle()
        wait(GRIP_WAIT)
        movel(P_ING3_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_APPROACH, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_ROT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        shake_ladle()
        movel(P_POUR_EXIT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING3_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING3_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        movel(P_ING3_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def lever_open():
        movej(J_LEVER_OPEN_NEUT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(posx(632.66, -244.13, 40.89, 151.83, -180.00, 149.67),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_close()
        wait(GRIP_WAIT)
        movej(posj(-21.75, 49.27, 42.50, -0.30, 88.23, 67.14),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        movej(J_LEVER_OPEN_NEUT, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def lever_close():
        movej(posj(-13.50, 0.37, 88.20, 24.61, 25.21, -28.38),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_LEVER_NEUTRAL, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(posx(632.66, -244.12, 40.89, 171.45, 180.00, -99.88),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_close()
        wait(GRIP_WAIT)
        movej(posj(-21.75, 49.27, 42.49, -0.30, 88.23, -23.68),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        movej(J_LEVER_NEUTRAL, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def scoop_stir():
        """힘제어 진입 후 실제 젓기 구간 (wok_final3 변경분).

        wok_final은 목표력 -5N → 10초 대기 → check_force_condition 원샷 체크(max=8) →
        move_spiral 이었다. wok_final3에서 아래로 바뀜:
          - 목표력 -10N (팬 바닥까지 확실히 눌러 넣기)
          - 원샷 체크 대신 0.5초 간격 폴링(최대 약 10초). 힘이 목표 시점보다 늦게 오르거나
            순간적으로 스쳐 지나가면 원샷 체크는 그대로 놓친다.
            (m0609_gear_force.py의 while True 폴링과 같은 패턴 + 무한대기 방지 타임아웃)
          - 판정 상한 8 → 15
          - 젓기: move_spiral → move_periodic(ref=DR_TOOL).
            move_spiral은 이 코드베이스에서 실동작이 검증된 적이 없다(axis=Z/ref가 실제
            삽입 방향과 안 맞으면 회전 평면이 팬 표면에 막혀 움직임이 사라짐).
            move_periodic은 shake_ladle/gear_force에서 이미 검증된 방식.
        """
        task_compliance_ctrl()
        set_stiffnessx([200.00, 200.00, 3000.00, 200.00, 200.00, 200.00], time=0.0)
        time.sleep(1.0)   # compliance 진입 완료 대기 (없으면 force가 무시됨)
        set_desired_force([0.00, 0.00, SCOOP_FORCE_N, 0.00, 0.00, 0.00],
                          [0, 0, 1, 0, 0, 0], time=0.0, mod=_DR_FC_MOD_REL)
        wait(10.00)

        # ★ ROS2는 0 = 조건 충족, -1 = 미충족 (DRL의 if와 의미가 반대라 == 0 비교)
        fc = -1
        for _ in range(SCOOP_FC_POLL_COUNT):
            fc = check_force_condition(axis=_DR_AXIS_Z, min=SCOOP_FC_MIN,
                                       max=SCOOP_FC_MAX, ref=_DR_BASE)
            tf = get_tool_force(ref=_DR_BASE)
            logger.info("scoop 힘제어: Fz=%s N, fc=%s (0=충족)"
                        % (tf[2] if tf != -1 else "?", fc))
            if fc == 0:
                break
            wait(SCOOP_FC_POLL_INTERVAL)

        if fc == 0:
            wait(0.3)   # check_force_condition 직후 바로 이어 보내면 무시될 수 있어 여유를 둠
            sp_ret = move_periodic(amp=[25.00, 25.00, 0.00, 0.00, 0.00, 0.00],
                                   period=[0.60, 0.40, 0.00, 0.00, 0.00, 0.00],
                                   atime=0.00, repeat=15, ref=DR_TOOL)
            logger.info("scoop 젓기(move_periodic) 반환값 ret=%s (0=성공 기대)" % sp_ret)
        else:
            logger.warn("scoop 힘 조건 %.0f초 내 미충족 — 젓기 스킵"
                        % (SCOOP_FC_POLL_COUNT * SCOOP_FC_POLL_INTERVAL))

        release_force(time=0.0)
        release_compliance_ctrl()

    def scoop():
        """힘제어 기반 국자 삽입 (wok_final3.py 기준).

        wok_final 대비 변경:
          - 주걱 파지 자세 J_SCOOP_GRIP 로 변경
          - 잡고 빼는 동작을 movel(P_SCOOP_TRANSIT_A) → movej(J_SCOOP_PULLOUT) 로 교체.
            P_SCOOP_TRANSIT_A는 복귀 경로에서만 사용된다.
          - 힘제어/젓기는 scoop_stir() 참고
        """
        movej(J_SCOOP_STANDBY, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_SCOOP_GRIP, radius=0.00, ra=DR_MV_RA_DUPLICATE)   # 주걱 잡기
        # 웍과 동일한 파지 폭 검증(56±2mm, 3회 재시도, 실패 시 비상정지)을 적용한다.
        # grip_close()가 하던 DO3 OFF(국자용 출력 중립화)는 공용 루틴이 DO1/DO2만 만지므로
        # 여기서 따로 해준다. 닫기 대기는 원래 값 2.00초를 그대로 유지(웍은 1.5초).
        set_digital_output(3, OFF)
        grip_and_verify_width('주걱', SCOOP_GRIP_WIDTH_TARGET_MM,
                              SCOOP_GRIP_WIDTH_TOLERANCE_MM, 2.00)
        movej(J_SCOOP_PULLOUT, radius=0.00, ra=DR_MV_RA_DUPLICATE)  # 잡고 빼는 위치
        movel(P_SCOOP_TRANSIT_B, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movej(J_SCOOP_TRANSIT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_SCOOP_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(posj(-3.88, 32.47, 49.41, 4.47, 60.80, 74.52),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)

        if SIMULATION:
            logger.warn("SIMULATION=True — scoop() 힘제어 구간 우회 (시뮬레이터엔 접촉 물리 없음)")
            wait(1.00)
        else:
            scoop_stir()

        movej(J_SCOOP_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_SCOOP_TRANSIT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(P_SCOOP_TRANSIT_B, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_SCOOP_TRANSIT_A, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(440.03, -355.20, 6.90, 133.30, -179.97, -133.95),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(2, ON)
        wait(GRIP_WAIT)
        movej(J_SCOOP_STANDBY, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def toss(count, label):
        """웍질 토스 반복. radius 블렌딩으로 이어지는 연속 동작이라 wait_motion 미적용
        (raw _movej 사용) — estop 체크포인트는 한 번의 웍질(toss) 완료 시점에만 둔다
        (wok_test4의 wokking()과 동일 granularity).

        ★ wok_final3 반영: 마지막 토스는 radius=0으로 완전히 멈춰서, 뒤이은
        movel(P_WOK_PLACE_VIA)가 블렌딩 관성 없이 진짜 직선으로 나가도록 한다 (화로 간섭 방지)."""
        for i in range(count):
            publish_stage("웍질 %s %d/%d" % (label, i + 1, count))
            last = (i == count - 1)
            _movej(J_TOSS_A, vel=sc_toss(75.62), acc=sc_toss(330.48),
                   radius=50.00, ra=DR_MV_RA_DUPLICATE)
            _movej(J_TOSS_B, vel=sc_toss(85.02), acc=sc_toss(408.33),
                   radius=(0.00 if last else 50.00), ra=DR_MV_RA_DUPLICATE)
            check_estop_and_wait()

    def grip_and_verify_width(label, target_mm, tolerance_mm, close_wait):
        """그리퍼를 닫고 파지 폭이 target±tolerance 범위인지 검증하는 공용 루틴.

        웍(프라이팬)과 주걱이 같은 조건·같은 실패 처리를 쓰도록 하나로 묶었다(요청 반영).
        따로 복사해두면 한쪽만 고쳤을 때 두 파지의 판정이 갈리기 때문.

        범위 밖이면 열었다 닫았다를 최대 WOK_GRIP_WIDTH_RETRY회 재시도하고, 그래도 범위
        밖이면 비상정지(물리 버튼과 동일)로 넘겨 사람이 직접 확인하게 한다.
        Modbus로 폭을 못 읽으면(연결 끊김) 검증을 건너뛰고 진행한다 — 조리를 멈추지
        않으려는 선택이라, 이때는 폭 검증이 통째로 무력화된다는 점에 유의.

        호출 전에 대상 위치로 이동해 두어야 한다(이 함수는 이동하지 않는다).
        """
        width = None
        for attempt in range(1, WOK_GRIP_WIDTH_RETRY + 1):
            set_digital_output(2, OFF)
            set_digital_output(1, ON)
            wait(close_wait)

            width = read_gripper_width_mm()
            if width is None:
                logger.warn('%s 파지 폭 읽기 실패(Modbus 연결 확인) — 폭 검증 없이 진행합니다' % label)
                publish_grip_result()
                return

            if abs(width - target_mm) <= tolerance_mm:
                logger.info('%s 파지 폭 정상 (%.1fmm, %d/%d회차)'
                            % (label, width, attempt, WOK_GRIP_WIDTH_RETRY))
                publish_grip_result()
                return

            logger.warn('%s 파지 폭 비정상 (%.1fmm, 목표 %.1f±%.1fmm, %d/%d회차)'
                        % (label, width, target_mm, tolerance_mm,
                           attempt, WOK_GRIP_WIDTH_RETRY))
            if attempt < WOK_GRIP_WIDTH_RETRY:
                set_digital_output(1, OFF)
                set_digital_output(2, ON)
                wait(GRIP_WAIT)

        trigger_estop_and_wait('%s 파지 폭 %d회 연속 비정상 (%.1fmm)'
                               % (label, WOK_GRIP_WIDTH_RETRY, width))

    def wok_pick():
        """웍(프라이팬) 파지 — 반드시 P_WOK_GRIP에서. 그리퍼 Modbus 파지검증(wok_test4 이식)에
        더해, 파지 폭이 WOK_GRIP_WIDTH_TARGET_MM±WOK_GRIP_WIDTH_TOLERANCE_MM(56±2mm) 범위인지
        검증한다 — 실제로 웍을 잘못(테두리가 아니라 헛)잡는 경우가 잦아서 추가한 예외처리."""
        movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        wait(0.50)
        grip_and_verify_width('웍', WOK_GRIP_WIDTH_TARGET_MM,
                              WOK_GRIP_WIDTH_TOLERANCE_MM, WOK_GRIP_WAIT)

    def wok_lower_to_stove():
        """웍을 화구(P_WOK_GRIP)로 내리는 공통 경로. ★ wok_final3 반영 — 직선으로 바로
        내려가면 화로에 걸려서 P_WOK_PLACE_VIA(Z=286)를 먼저 거친다. 그리퍼는 건드리지
        않으므로 '내려놓기'(wok_place)와 '잡은 채 휴지' 양쪽에서 재사용한다."""
        movel(P_WOK_PLACE_VIA, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def wok_place():
        """웍 내려놓기 — 파지 좌표와 동일한 P_WOK_GRIP으로 통일. 경유점 경로로 내려간 뒤 개방."""
        wok_lower_to_stove()
        set_digital_output(1, OFF)
        set_digital_output(2, ON)
        wait(GRIP_WAIT)

    def pour_to_bowl(lift_after=False):
        """웍을 들고 그릇에 붓기. wok_final3에서 볶음밥 붓기 좌표가 부침개와 같아져
        (413.20, 176.55, 392.31, ...) → (393.35, 208.74, 398.03, ...) 두 조리가 공용한다.
        유일한 차이는 붓고 나서 위로 70mm 빼는 동작으로, probe_flip3(부침개)에만 있다."""
        movel(P_POUR_LIFTOFF, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movej(J_POUR_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_BOWL, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        if lift_after:
            movel(posx(0.00, 0.00, 70.00, 0.00, 0.00, 0.00),
                  radius=0.00, ref=0, mod=DR_MV_MOD_REL, ra=DR_MV_RA_DUPLICATE)
        movej(J_POUR_RETRACT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_POUR_RETURN, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def wokking():
        """준비 단계(레버+웍파지)가 끝난 상태에서 시작. 1차 웍질→scoop→재파지→본 웍질→붓기→원위치."""
        _movej(J_TOSS_A, vel=sc_toss(75.62), acc=sc_toss(330.48),
               radius=50.00, ra=DR_MV_RA_DUPLICATE)
        toss(WOK_INNER, "1차")

        wok_place()
        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)

        publish_stage("scoop 시작")
        scoop()
        publish_stage("scoop 완료")

        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        wok_pick()

        for set_i in range(WOK_OUTER):
            publish_stage("웍질 세트 %d/%d" % (set_i + 1, WOK_OUTER))
            toss(WOK_INNER, "세트%d" % (set_i + 1))
            # 웍을 잡은 채 화구 원위치에서 휴지. 경유점 경로(wok_final3)로 내려간다.
            wok_lower_to_stove()
            wait(SET_REST_WAIT)

        pour_to_bowl(lift_after=False)

        wok_place()

    # ── 부침개(jeon) 서브루틴 (probe_flip3.py 기준) ────────────────────────

    def jeon_flip():
        """뒤집기 반복. 던지고(FLAP) 받는(FLAT) 사이는 원본(probe_flip3)에서 50ms 로
        정교하게 튜닝된 타이밍이라, toss()와 동일하게 raw _movej/_movel 사용 + estop
        체크포인트는 한 뒤집기 사이클(READY→FLAP→FLAT→착지) 완료 시점에 한 번만 둔다.
        공통 movel/movej 래퍼(서비스 호출+Modbus 폭 읽기)를 매 서브동작마다 통과시키면
        이 타이밍이 깨져 던지고 받는 동작 자체가 어긋날 수 있다."""
        for i in range(FLIP_COUNT):
            publish_stage("뒤집기 %d/%d" % (i + 1, FLIP_COUNT))
            _movej(READY_J, vel=VELOCITY_J, acc=ACC_J, radius=0, ra=DR_MV_RA_DUPLICATE)
            wait(0.3)
            _movej(FLAP_J, vel=WOK_VEL_J, acc=WOK_ACC_J, radius=0, ra=DR_MV_RA_DUPLICATE)
            wait(0.05)
            _movej(FLAT_J, vel=WOK_VEL_J, acc=WOK_ACC_J, radius=0, ra=DR_MV_RA_DUPLICATE)
            wait(0.1)
            _movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
            check_estop_and_wait()
            if i < FLIP_COUNT - 1:
                wait(FLIP_SET_REST_WAIT)

    # ── 준비 단계 정의 ──────────────────────────────────────────────────

    def stage_home_and_ingredients1():
        movej(J_HOME, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        set_velx(250.0 * INGREDIENT_SPEED_RATIO, 80.625 * INGREDIENT_SPEED_RATIO)
        set_accx(1000.0 * INGREDIENT_SPEED_RATIO, 322.5 * INGREDIENT_SPEED_RATIO)
        publish_stage("재료 투입 1/3")
        ingredients1()

    def stage_ingredients2():
        publish_stage("재료 투입 2/3")
        ingredients2()

    def stage_ingredients3():
        publish_stage("재료 투입 3/3")
        ingredients3()
        set_velx(250.0 * r, 80.625 * r)
        set_accx(1000.0 * r, 322.5 * r)
        wait(1.00)

    def stage_lever_and_grip():
        publish_stage("레버 열기")
        lever_open()
        publish_stage("웍 파지")
        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        wok_pick()

    def stage_jeon_home_and_ingredients():
        publish_stage("홈 복귀")
        movej(J_HOME, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        set_velx(250.0 * INGREDIENT_SPEED_RATIO, 80.625 * INGREDIENT_SPEED_RATIO)
        set_accx(1000.0 * INGREDIENT_SPEED_RATIO, 322.5 * INGREDIENT_SPEED_RATIO)
        publish_stage("재료 투입(반죽)")
        ingredients2()
        set_velx(250.0 * FLIP_SPEED_RATIO, 80.625 * FLIP_SPEED_RATIO)
        set_accx(1000.0 * FLIP_SPEED_RATIO, 322.5 * FLIP_SPEED_RATIO)

    def stage_jeon_lever_and_grip():
        movel(posx(544.14, -498.37, 292.14, 1.33, 93.74, 89.43),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        publish_stage("레버 열기")
        lever_open()
        set_digital_output(2, OFF)   # probe_flip3와 동일 — 웍 파지 전 출력 중립화
        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        publish_stage("웍 파지")
        wok_pick()

    PREP_STAGES_FRIED_RICE = [
        ('홈복귀+재료투입1', stage_home_and_ingredients1),
        ('재료투입2', stage_ingredients2),
        ('재료투입3', stage_ingredients3),
        ('레버열기+웍파지', stage_lever_and_grip),
    ]
    PREP_STAGES_JEON = [
        ('홈복귀+재료투입', stage_jeon_home_and_ingredients),
        ('레버열기+웍파지', stage_jeon_lever_and_grip),
    ]

    # ── 메인 시퀀스: dish 파라미터로 분기 ───────────────────────────────

    if dish == 'fried_rice':
        set_velj(60.0 * r)
        set_accj(100.0 * r)
        set_velx(250.0 * r, 80.625 * r)
        set_accx(1000.0 * r, 322.5 * r)

        run_preparation(PREP_STAGES_FRIED_RICE)

        publish_stage("웍질 시작")
        wokking()

        movel(P_POST_RELEASE, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

        publish_stage("레버 닫기")
        lever_close()

        movej(J_HOME, radius=100.00, ra=DR_MV_RA_DUPLICATE)

    else:  # jeon
        fr = FLIP_SPEED_RATIO
        set_velj(60.0 * fr)
        set_accj(100.0 * fr)
        set_velx(250.0 * fr, 80.625 * fr)
        set_accx(1000.0 * fr, 322.5 * fr)

        run_preparation(PREP_STAGES_JEON)

        publish_stage("뒤집기 시작")
        jeon_flip()

        publish_stage("그릇에 붓기")
        pour_to_bowl(lift_after=True)

        publish_stage("웍 내려놓기")
        wok_place()

        movel(P_POST_RELEASE, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

        publish_stage("레버 닫기")
        lever_close()

        movej(J_HOME, radius=30.00, ra=DR_MV_RA_DUPLICATE)

    publish_stage("완료")
    gripper_client.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
