"""wok_integrate5 — wok_integrate4의 "그리퍼 파지 실패 → 사람 개입 → 재개" 경로를 고친 버전.

동작(레시피/좌표/속도/힘제어)은 wok_integrate4와 100% 동일하다. 바뀐 것은 예외처리뿐이다.

■ v4에서 실제로 일어난 일 (이 파일이 고치는 문제)
  웍 파지 폭 검증이 3회 실패 → 정지 → 사람이 DI15(열기)/DI14(닫기)로 그리퍼가 웍을 잡게
  해줌 → UI "작업 재개" → 로봇이 말도 안 되는 행동을 함.

  원인은 v4의 grip_and_verify_width() 실패 처리에 있다:
    trigger_estop_and_wait() → 재개되면 _in_preparation이 True라 RestartPreparation을 던지고,
    run_preparation()이 '레버열기+웍파지' 단계를 처음부터 다시 돌린다. 그 단계의 첫 동작이
    lever_open() 인데, 이때 그리퍼에는 사람이 쥐어준 웍이 들려 있다:
      1) movej(J_LEVER_OPEN_NEUT)  — 웍을 든 채 레버 쪽으로 끌고 나감
      2) movel(632.66,-244.13,Z=40.89) — 웍을 든 채 레버 높이로 하강 → 충돌
      3) grip_open()               — 레버를 놓는 동작인데 실제로는 웍을 그 자리에 떨어뜨림
      4) 이어지는 wok_pick()이 빈 화구를 잡음 → 또 3회 실패 → 다시 정지 (무한 루프)
  즉 "재시작할 단계"가 "지금 그리퍼에 뭘 들고 있는지"를 전혀 모르는 것이 근본 원인이다.

■ 고친 내용
  1) 재개 직후 사람이 만들어준 파지를 '로봇이 움직이기 전에' 재검증한다.
     통과(56±2mm & grip_detected=True)하면 재파지도 단계 재시작도 하지 않고 그 자리에서
     다음 동작으로 이어간다 — 사람이 이미 웍을 물려줬는데 lever_open()을 다시 도는 것은
     무의미할 뿐 아니라 위험하기 때문. (v4의 1~4번 경로가 통째로 사라진다)
  2) 파지물 추적 _held. 단계를 처음부터 다시 돌리기로 했을 때는 반드시
     safe_release_and_home()을 먼저 거친다 — 들고 있는 것(웍/레버/국자)을 제자리에
     내려놓고 J_WOK_APPROACH→J_HOME으로 빠져나온 뒤에야 재시작한다.
     ★ v4에서 '웍을 든 채 stage 재시작'이 나던 경로를 원천 차단한다.
  3) '레버열기+웍파지'를 '레버열기' / '웍파지' 두 단계로 분리. 파지 실패가 레버 동작을
     다시 끌고 오지 않는다.
  4) 모션을 내보내기 '전에' 로봇 상태를 확인하는 ensure_movable(). v4는 모션 완료 후에만
     확인해서, 재개 직후 첫 모션이 SAFE_STOP 상태로 나가는 것을 못 막았다.
     (매 모션마다 하면 서비스 호출이 두 배가 되므로 '재개 직후 1회'에만 건다)
  5) 재개 후 실제 TCP가 기대 위치에서 POSE_TOLERANCE_MM 이상 벗어나 있으면 movel로
     직행하지 않고 알려진 조인트 자세(J_WOK_APPROACH)를 먼저 거친다. 사람이 조그로
     로봇을 옮겼을 때 직선 이동이 그대로 나가는 것을 막는다.
  6) 사람 개입 대기에 들어갈 때 그리퍼를 열고 DO1/DO2/DO3를 전부 OFF(중립)로 둔다.
     ★ DI14/DI15는 estop_button_io_integrate가 Modbus로 그리퍼 폭을 직접 움직이는데
       (닫기 0mm / 열기 110mm, 20N), 이쪽 DO가 반대 명령을 물고 있으면 서로 싸운다.
       재개해서 파지가 확인되면 그때 DO1을 다시 ON으로 못 박는다(resync_gripper_do).
  7) 웍질 중(_in_preparation=False) 파지 실패도 절대 그냥 통과시키지 않는다. v4는
     trigger_estop_and_wait()가 그냥 return해서 검증 없이 toss()를 TOSS_SPEED_RATIO=1.25
     고속으로 실행했다. v5는 wok_pick()이 bool을 반환하고, 확보 못 하면 웍질에 진입하지 않는다.
  8) 재개 신호를 UI(/estop=False)와 DI16(/manual_resume) 양쪽에서 받는다.
     v4는 /estop만 봤는데, DI16은 recovery_manager에서 manual_estop_pending이 False라
     무시되어 /reset_estop이 안 나갔다 → 물리 재개 버튼으로는 이 정지가 안 풀렸다.
  9) '폭 읽기 실패 → 검증 없이 낙관적 진행'을 연속 MODBUS_FAIL_LIMIT회로 제한. v4는
     무제한이라 Modbus가 끊기면 폭 검증이 통째로 무력화된 채 조리가 계속됐다.
 10) 조리 중에는 /gripper_state 로 'GRIP_FAILED'를 발행하지 않는다.
     ★ 발행하면 safety_monitor → /alarm GRIP_FAILED → recovery_manager → /reset_robot 'HOME'
       → robot_command_bridge가 '이 노드와 동시에' 로봇을 홈으로 movej 한다(이중 제어).
       대시보드에는 GRIP_RETRY로만 알린다. PUBLISH_GRIP_FAILED_STATE 참고.

■ 사람이 해야 하는 일 (파지 실패로 정지했을 때)
     DI15(열기) → 그리퍼에 웍을 끼운다 → DI14(닫기) → UI "작업 재개" 또는 DI16
  재개 후 로봇은 움직이기 전에 폭/파지감지를 다시 읽어보고,
     정상이면 → 그 자리에서 하던 작업을 이어감
     비정상이면 → 로봇이 스스로 다시 파지를 시도(최대 WOK_PICK_CYCLE_MAX회)

ROS 파라미터 `dish`로 조리를 선택한다 (기본값 fried_rice):
    ros2 run rokey wok_integrate5 --ros-args -p dish:=fried_rice
    ros2 run rokey wok_integrate5 --ros-args -p dish:=jeon

┌─ fried_rice (wok_final3 기준) ───────────────────────────────────────────┐
│ 재료 3종 투입 → 레버 열기 → 웍 파지 → 1차 웍질 → scoop(힘제어 국자삽입)   │
│ → 웍 재파지 → 본 웍질(WOK_OUTER×WOK_INNER) → 붓기 → 웍 원위치 → 레버 닫기 │
└───────────────────────────────────────────────────────────────────────────┘
┌─ jeon (probe_flip3 기준) ──────────────────────────────────────────────────┐
│ 재료 투입(반죽, ingredients2 재사용) → 레버 열기 → 웍 파지 → flip 반복     │
│ (FLIP_COUNT) → 그릇에 붓기 → 웍 원위치 → 레버 닫기                        │
└───────────────────────────────────────────────────────────────────────────┘
"""
import math
import time

import rclpy
from rclpy.logging import get_logger
from std_msgs.msg import String, Bool, Empty, Float32
from geometry_msgs.msg import Point
from dsr_msgs2.srv import GetRobotState, SetRobotControl
from rokey.onrobot_rg2 import RG2Client, RG2Error, UnitId

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# ▼▼▼ 볶음밥(fried_rice) 파라미터 — wok_final3.py 기준 (v4와 동일) ▼▼▼
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

# ▼▼▼ scoop 힘제어 파라미터 — wok_final3에서 상향된 값 (v4와 동일) ▼▼▼
SCOOP_FORCE_N = -10.00        # 목표력 Z
SCOOP_FC_MIN = 2              # check_force_condition 하한
SCOOP_FC_MAX = 15             # 상한
SCOOP_FC_POLL_COUNT = 20      # 0.5초 간격 20회 = 약 10초 폴링
SCOOP_FC_POLL_INTERVAL = 0.5
# ▲▲▲

# ▼▼▼ 파지 폭 검증 ▼▼▼
# 정상 파지 시 그리퍼 폭은 56mm 근방이어야 한다. 벗어나면 웍 테두리를 잘못 물었거나 헛잡은 것.
WOK_GRIP_WIDTH_TARGET_MM = 56.0
WOK_GRIP_WIDTH_TOLERANCE_MM = 2.0
WOK_GRIP_WIDTH_RETRY = 3

# 주걱(국자) 파지 폭 — 웍과 같은 조건. 손잡이 두께가 다르면 실측해서 여기만 조정.
SCOOP_GRIP_WIDTH_TARGET_MM = 56.0
SCOOP_GRIP_WIDTH_TOLERANCE_MM = 2.0

# 한 '사이클' = 로봇이 WOK_GRIP_WIDTH_RETRY회 재시도 → 실패 → 사람 개입 대기 → 재개.
# 이 사이클을 몇 번까지 반복할지. 소진하면 준비 단계에서는 안전 정리 후 단계 재시작,
# 조리 중이면 웍질에 진입하지 않고 안전하게 빠져나온다.
WOK_PICK_CYCLE_MAX = 3

# 그리퍼 폭을 Modbus로 못 읽을 때 '검증 없이 진행'을 허용하는 연속 횟수. 이걸 넘으면
# 검증 불능 상태로 조리를 계속하지 않고 사람을 부른다(v4는 무제한이었다).
MODBUS_FAIL_LIMIT = 3
# ▲▲▲

# ▼▼▼ v5 신규: 사람 개입 후 재개 안전장치 ▼▼▼
# 재개 후 실제 TCP가 기대 위치에서 이만큼 벗어나 있으면 movel 직행을 포기하고
# 알려진 조인트 자세를 먼저 거친다 (사람이 조그로 로봇을 옮겼을 가능성).
POSE_TOLERANCE_MM = 30.0
# 안전 정리(safe_release_and_home) 구간 속도 배율. 위치가 불확실한 상태에서 움직이므로 낮춘다.
RECOVERY_SPEED_RATIO = 0.3
# ★ 조리 노드가 로봇을 쥐고 있는 동안 /gripper_state로 'GRIP_FAILED'를 발행하면
#   safety_monitor → /alarm GRIP_FAILED → recovery_manager → /reset_robot 'HOME' →
#   robot_command_bridge가 '동시에' 로봇을 홈으로 movej 한다(이중 제어).
#   켜지 말 것. 대시보드에는 GRIP_RETRY로 알린다.
PUBLISH_GRIP_FAILED_STATE = False
# ▲▲▲

# ▼▼▼ 부침개(jeon) 파라미터 — probe_flip3.py 기준 (v4와 동일) ▼▼▼
FLIP_SPEED_RATIO = 0.6
FLIP_COUNT = 3
FLIP_SET_REST_WAIT = 3.00
VELOCITY_J = 10                 # ready_joint 진입 전용
ACC_J = 20
WOK_VEL_J = 500                 # flap/flat(던지고 받기) 전용
WOK_ACC_J = 1000
READY_JOINT = [3.01, 8.44, 53.86, -2.55, 72.76, 3.06]
FLAP_JOINT = [3.01, 5.0, 43.00, -2.55, 45, 3.06]
FLAT_JOINT = [3.01, 5.0, 43.00, -2.55, 50, 3.06]
# ▲▲▲

WAIT_MOTION = 0.50   # movel/movej 사이 공통 wait_motion. 연속 블렌딩 구간(toss)만 제외.

# 그리퍼(OnRobot RG2) Compute Box — Modbus TCP 직결.
# ★ estop_button_io_integrate도 같은 Compute Box에 자체 RG2Client로 붙는다(DI14/15용).
#   폭 읽기가 계속 실패하면 이 동시 연결을 먼저 의심할 것 — MODBUS_FAIL_LIMIT에 걸려
#   조리가 멈추면 그게 원인일 가능성이 높다.
GRIPPER_MODBUS_HOST = "192.168.1.1"
GRIPPER_MODBUS_UNIT_ID = UnitId.QUICK_CHANGER

# ─────────────────────────────────────────────────────────────────────────
# 반복 좌표 상수 (wok_final3.py 기준, v4와 동일). main() 안에서 posj()/posx()로 래핑된다.
# ─────────────────────────────────────────────────────────────────────────
_J_HOME             = [0.00, 0.00, 90.00, 0.00, 90.00, 0.00]
_J_LEVER_NEUTRAL    = [-21.56, 45.45, 31.34, -0.05, 99.96, -16.28]
_J_LEVER_OPEN_NEUT  = [-20.55, 43.31, 29.64, -0.39, 106.07, -23.54]
_J_WOK_APPROACH     = [-4.10, -9.12, 133.07, -1.39, -17.77, -0.13]
_J_TOSS_A           = [-3.40, 8.62, 81.29, -0.28, 42.61, -0.05]
_J_TOSS_B           = [-3.34, -3.85, 81.09, -0.24, 55.05, -0.05]
_J_SCOOP_STANDBY    = [-39.77, 28.57, 55.08, -0.24, 96.06, 53.34]
_J_SCOOP_TRANSIT    = [-47.32, 24.05, 36.21, -0.33, 119.71, 45.99]
_J_SCOOP_APPROACH   = [-4.11, 23.42, 36.90, 8.99, 67.66, 74.51]
_J_SCOOP_GRIP       = [-40.11, 35.35, 73.83, 0.66, 73.86, 51.94]
_J_SCOOP_PULLOUT    = [-39.84, 31.64, 73.83, 0.36, 75.10, 51.94]

# ★ 웍 파지 = 웍 내려놓기 = probe_flip의 stove_pos, 전부 이 좌표로 통일.
_P_WOK_GRIP         = [704.84, -21.97, 181.70, 178.01, -109.75, 178.85]
_P_WOK_PLACE_VIA    = [703.08, -21.33, 286.00, 177.81, -114.16, 179.21]
_P_POST_RELEASE     = [611.20, -15.91, 214.31, 178.09, -111.38, 178.50]

_P_POUR_APPROACH    = [903.68, -48.07, 291.30, 31.02, 98.82, 93.12]
_P_POUR_ROT         = [903.68, -48.06, 291.30, 31.02, 98.82, -59.87]
_P_POUR_EXIT        = [904.95, -165.04, 270.56, 24.62, 98.82, 93.11]

_P_ING1_STANDBY     = [517.78, -645.92, 80.11, 0.86, 94.19, 88.94]
_P_ING1_GRIP        = [615.78, -642.28, 73.26, 0.73, 94.19, 88.74]
_P_ING1_LIFT        = [620.84, -582.39, 378.66, 2.92, 93.06, 88.11]   # v4 복귀 경로의 상단 경유점
_P_ING2_LIFT        = [620.73, -528.87, 370.86, 1.12, 94.10, 88.96]
_P_ING2_RELEASE     = [623.80, -528.80, 72.11, 1.49, 93.96, 89.23]
_P_ING2_STANDBY     = [532.01, -532.66, 79.35, 1.37, 94.09, 89.43]
_P_ING3_STANDBY     = [507.39, -433.03, 78.81, 1.59, 94.27, 89.62]
_P_ING3_GRIP        = [614.61, -423.47, 73.75, 1.43, 94.14, 89.23]
_P_ING3_LIFT        = [614.61, -423.47, 370.75, 1.43, 94.14, 89.23]

_P_SCOOP_TRANSIT_A  = [439.91, -356.18, 26.75, 136.39, -179.80, -130.78]  # (1회, 복귀 경로)
_P_SCOOP_TRANSIT_B  = [335.25, -356.98, 32.47, 168.59, 179.74, -98.19]    # (2회)
_P_SCOOP_RELEASE    = [440.03, -355.20, 6.90, 133.30, -179.97, -133.95]   # 주걱 놓는 위치

_P_POUR_LIFTOFF     = [698.57, -14.97, 330.02, 178.04, -117.09, 178.50]
_P_POUR_BOWL        = [393.35, 208.74, 398.03, 24.66, 101.59, -137.26]
_J_POUR_APPROACH    = [11.32, -44.41, 118.37, 3.22, 36.75, -7.76]
_J_POUR_RETRACT     = [11.56, -39.43, 124.18, 3.22, 25.94, -7.76]
_J_POUR_RETURN      = [-3.94, 3.44, 101.11, 3.22, 7.08, -7.76]

# 레버 관련 좌표 (v4에서는 함수 안에 인라인이었으나, safe_release_and_home이 참조해야 해서 상수화)
_P_LEVER_OPEN_GRIP  = [632.66, -244.13, 40.89, 151.83, -180.00, 149.67]
_J_LEVER_OPEN_PULL  = [-21.75, 49.27, 42.50, -0.30, 88.23, 67.14]
_P_LEVER_CLOSE_GRIP = [632.66, -244.12, 40.89, 171.45, 180.00, -99.88]
_J_LEVER_CLOSE_PULL = [-21.75, 49.27, 42.49, -0.30, 88.23, -23.68]
_J_LEVER_CLOSE_ENTRY = [-13.50, 0.37, 88.20, 24.61, 25.21, -28.38]

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
logger = get_logger("wok_integrate5")


class RestartPreparation(Exception):
    """준비 단계 도중 안전정지가 걸리면 이 예외를 던져 그 단계만 처음부터 다시 시작한다.
    중간부터 이어가면 재료/파지가 실제로 제대로 됐는지 검증할 방법이 없기 때문.

    ★ v5: 이 예외를 던지기 전에는 반드시 safe_release_and_home()을 거친다.
      v4는 그리퍼에 웍을 든 채로 단계를 재시작해서 lever_open()이 웍을 끌고 나갔다."""
    pass


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("wok_integrate5", namespace=ROBOT_ID)
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
    # v5 신규 상태 ---------------------------------------------------------
    _resume_req = {'pending': False}   # DI16(/manual_resume) 수신 래치
    _held = {'item': None}             # 그리퍼가 지금 들고 있는 것 (safe_release_and_home이 참조)
    _recovering = {'active': False}    # 안전 정리 중 — 재귀적 복구/단계재시작 금지
    _resume = {'dirty': False}         # 재개 직후 1회: 모션 전 로봇 상태 확인 필요
    _modbus_fail = {'count': 0}        # 폭 읽기 연속 실패 횟수
    _speed = {'base': SPEED_RATIO}     # 현재 조리의 기준 속도 배율 (복구 후 원복용)

    # ── 관리자 대시보드 실시간 연동 (topic 이름은 v4와 동일하게 유지) ──
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

    def on_manual_resume(_msg):
        """DI16(물리 재개 버튼) → estop_button_io_integrate가 /manual_resume을 발행한다.
        ★ v5 신규. v4는 /estop만 봤는데, recovery_manager는 manual_estop_pending이 False면
        DI16을 무시해서 /reset_estop을 안 내보낸다 — 즉 파지 실패 정지는 물리 재개 버튼으로
        풀리지 않았고 UI "작업 재개"(/estop=False 직접 발행)로만 풀렸다. 여기서 직접 받는다."""
        _resume_req['pending'] = True
        logger.warn('/manual_resume(DI16) 수신 — 재개 요청')

    node.create_subscription(Bool, '/estop', on_estop, 10)
    node.create_subscription(Empty, '/manual_resume', on_manual_resume, 10)

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

    # safety_monitor.gripper_state_callback()이 자동복구 알람으로 바꾸는 상태값들.
    #   'GRIP_FAILED'/'GRIP_UNSTABLE'      → /alarm 동일 코드
    #   'GRIPPED' 다음의 'NOT_GRIPPED'/'GRIP_LOST' → /alarm PAN_DROPPED
    # 셋 다 recovery_manager가 /reset_robot 'HOME'을 보내고, robot_command_bridge가
    # '이 노드와 동시에' 로봇을 홈으로 movej 한다 — 조리 노드가 로봇을 쥐고 있는 동안
    # 다른 프로세스가 같은 로봇에 모션을 내보내는 이중 제어다. 조리 중에는 발행 금지.
    _ALARMING_GRIP_STATES = ('GRIP_FAILED', 'GRIP_UNSTABLE', 'NOT_GRIPPED', 'GRIP_LOST')

    def publish_grip_state(state):
        """/gripper_state 발행 게이트 (★ v5 신규).

        위 _ALARMING_GRIP_STATES는 대시보드용 표시값으로 바꿔서 내보낸다. safety_monitor의
        어느 분기에도 안 걸리는 코드라 자동복구(=로봇 이중 제어)를 유발하지 않는다.
        정상적인 '놓기'(웍 내려놓기 등)까지 PAN_DROPPED로 잡히는 것도 여기서 막힌다."""
        if state in _ALARMING_GRIP_STATES and not PUBLISH_GRIP_FAILED_STATE:
            mapped = 'GRIP_RETRY' if state in ('GRIP_FAILED', 'GRIP_UNSTABLE') else 'RELEASED'
            gripper_state_pub.publish(String(data=mapped))
            return
        gripper_state_pub.publish(String(data=state))

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

    # ── 파지 판정 ────────────────────────────────────────────────────────

    def check_grip(target_mm, tolerance_mm):
        """(ok, width, detected) 반환. ok가 None이면 '판정 불가'(Modbus 읽기 실패).

        폭 하나만 보던 v4와 달리 grip_detected(하드웨어 파지감지)도 함께 본다 —
        폭이 우연히 맞아도 실제로 뭔가에 걸려 멈춘 게 아니면 파지가 아니다."""
        width = read_gripper_width_mm()
        detected = read_gripper_grip_detected()
        if width is None:
            return None, None, detected
        ok = (abs(width - target_mm) <= tolerance_mm) and (detected is not False)
        return ok, width, detected

    def describe_grip(width, detected):
        w = f'{width:.1f}mm' if width is not None else '읽기실패'
        d = {True: 'True', False: 'False', None: '읽기실패'}[detected]
        return f'폭={w}, grip_detected={d}'

    # ── 로봇 안전정지 상태 폴링 (토픽 전달 타이밍에 의존 안 함) ──────────
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

    def ensure_movable():
        """★ v5 신규 — 모션을 '내보내기 전에' 로봇이 움직일 수 있는 상태인지 확인한다.

        v4는 check_estop_and_wait()를 모션 '완료 후'에만 걸어서, 재개 직후 첫 모션이
        SAFE_STOP 상태로 나가는 것을 막지 못했다. 매 모션마다 걸면 get_robot_state
        서비스 호출이 두 배가 되므로, 실제로 위험한 '재개 직후 1회'에만 건다."""
        state = read_robot_state()
        if state is None or state not in STATE_NEEDS_CLEAR:
            return
        logger.warn(f'모션 직전 로봇 상태 비정상 (robot_state={state}) — 해제 후 진행합니다')
        clear_safe_stop_until_ready()

    def wait_for_external_resume():
        """UI "작업 재개"(/estop=False) 또는 DI16(/manual_resume) 중 먼저 오는 쪽으로 풀린다.

        ★ v5: DI16 경로가 v4에는 없었다. DI16 → estop_button_io_integrate → /manual_resume →
        recovery_manager는 manual_estop_pending(=DI13으로 눌린 경우)이 아니면 무시해버려
        /reset_estop이 안 나갔고, 결과적으로 물리 재개 버튼으로는 이 정지가 안 풀렸다.
        여기서 /manual_resume을 직접 받고, 다른 노드들도 해제를 알 수 있도록
        /reset_estop을 대신 발행해 safety_monitor가 /estop=False를 내려주게 한다."""
        _resume_req['pending'] = False
        logger.error('E-STOP 활성 — 자동복구 안 함. 외부 해제(재개)까지 대기합니다.')
        logger.error('  ▶ UI "작업 재개" 또는 물리 DI16 버튼을 눌러주세요.')
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if not _estop['active']:
                logger.info('재개 확인 (/estop=False)')
                break
            if _resume_req['pending']:
                logger.info('재개 확인 (/manual_resume, DI16)')
                _estop['active'] = False
                reset_estop_pub.publish(Empty())
                break
        _resume_req['pending'] = False
        _resume['dirty'] = True

    def check_estop_and_wait():
        """movel/movej 완료 직후마다 호출. A) 로봇이 실제 안전정지 상태면 3초 대기→해제→
        준비 단계 중이면 그 단계 재시작. B) 로봇은 정상인데 /estop만 걸려 있으면 외부
        해제까지 대기 후 재시작.

        ★ v5: 단계 재시작(RestartPreparation) 전에 safe_release_and_home()을 반드시 거친다.
        v4는 그리퍼에 웍/레버/국자를 든 채로 단계를 처음부터 돌려서, 다음 단계의 첫 모션이
        들고 있는 물건과 함께 엉뚱한 곳으로 날아갔다."""
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
            _resume['dirty'] = True

            restart_preparation_if_needed()

        elif _estop['active']:
            wait_for_external_resume()
            restart_preparation_if_needed()

    def restart_preparation_if_needed():
        """준비 단계 중이면 안전 정리 후 그 단계를 처음부터 다시 시작한다.
        _recovering 중이면 아무것도 하지 않는다 — 안전 정리 자체가 movel/movej를 쓰고
        그 안에서 다시 이 함수가 불리므로, 재귀와 정리-중-재시작을 막아야 한다."""
        if not _in_preparation['active'] or _recovering['active']:
            return
        safe_release_and_home()
        raise RestartPreparation()

    def trigger_estop_and_wait(reason, alarm='GRIP_WIDTH_ABNORMAL'):
        """로봇 자체는 정상이지만 사람이 직접 확인해야 하는 상황(파지 폭 이상 등)에서
        물리 비상정지와 동일하게 /estop을 발행하고 외부 재개까지 대기한다.

        ★ 알람 코드를 'GRIP_WIDTH_ABNORMAL'로 두는 것은 의도적이다. 'GRIP_FAILED'로 바꾸면
        recovery_manager가 자동복구에 들어가 /reset_robot 'HOME'을 보내고,
        robot_command_bridge가 이 노드와 동시에 로봇을 홈으로 움직인다(이중 제어)."""
        if not _estop['active']:
            logger.error(f'{reason} — 비상정지 발행')
            _estop['active'] = True
            alarm_pub.publish(String(data=alarm))
            estop_pub.publish(Bool(data=True))
        wait_for_external_resume()

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
                logger.warn(f'"{name}" 단계 중 정지 발생 — 안전 정리 후 이 단계부터 다시 시작합니다.')
                continue
        _in_preparation['active'] = False

    # ─────────────────────────────────────────────────────────────────────
    # DSR_ROBOT2 import (v4와 동일)
    # ─────────────────────────────────────────────────────────────────────
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

    # 힘제어 상수는 설치 버전에 따라 없을 수 있어 방어적으로 처리.
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

    # DSR_ROBOT2는 import 시점에 모든 서비스 클라이언트를 한꺼번에 생성한다. 서버가 이미
    # 떠 있어도 '이 클라이언트'는 방금 생성된 것이라 DDS 매칭이 아직 안 끝났을 수 있고,
    # 매칭 전에 call_async()가 나가면 기본 QoS(RELIABLE+VOLATILE)라 요청이 재전송 없이
    # 유실되어 future가 영원히 안 풀린다 — set_singular_handling(DR_AVOID)이 매번 같은
    # 지점에서 멈추던 원인. 첫 모션 호출 전에 매칭이 끝나도록 짧게 대기한다.
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(1.0)

    # posj/posx 래핑 ---------------------------------------------------------
    J_HOME            = posj(_J_HOME)
    J_LEVER_NEUTRAL   = posj(_J_LEVER_NEUTRAL)
    J_LEVER_OPEN_NEUT = posj(_J_LEVER_OPEN_NEUT)
    J_LEVER_OPEN_PULL = posj(_J_LEVER_OPEN_PULL)
    J_LEVER_CLOSE_PULL = posj(_J_LEVER_CLOSE_PULL)
    J_LEVER_CLOSE_ENTRY = posj(_J_LEVER_CLOSE_ENTRY)
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
    P_ING1_LIFT       = posx(_P_ING1_LIFT)
    P_ING2_LIFT       = posx(_P_ING2_LIFT)
    P_ING2_RELEASE    = posx(_P_ING2_RELEASE)
    P_ING2_STANDBY    = posx(_P_ING2_STANDBY)
    P_ING3_STANDBY    = posx(_P_ING3_STANDBY)
    P_ING3_GRIP       = posx(_P_ING3_GRIP)
    P_ING3_LIFT       = posx(_P_ING3_LIFT)
    P_SCOOP_TRANSIT_A = posx(_P_SCOOP_TRANSIT_A)
    P_SCOOP_TRANSIT_B = posx(_P_SCOOP_TRANSIT_B)
    P_SCOOP_RELEASE   = posx(_P_SCOOP_RELEASE)
    P_POUR_LIFTOFF    = posx(_P_POUR_LIFTOFF)
    P_POUR_BOWL       = posx(_P_POUR_BOWL)
    P_LEVER_OPEN_GRIP = posx(_P_LEVER_OPEN_GRIP)
    P_LEVER_CLOSE_GRIP = posx(_P_LEVER_CLOSE_GRIP)

    READY_J = posj(*READY_JOINT)
    FLAP_J = posj(*FLAP_JOINT)
    FLAT_J = posj(*FLAT_JOINT)

    # 공통 래퍼: 모든 movel/movej는 모션 완료 후 estop 체크포인트를 거친다
    # (toss()/jeon_flip()의 연속 블렌딩 구간만 raw _movel/_movej로 예외).
    # ★ v5: 재개 직후(_resume['dirty'])에는 모션을 내보내기 '전에' 로봇 상태도 확인한다.
    def movel(*a, **kw):
        if _resume['dirty']:
            _resume['dirty'] = False
            ensure_movable()
        _movel(*a, **kw)
        mwait(time=WAIT_MOTION)
        check_estop_and_wait()

    def movej(*a, **kw):
        if _resume['dirty']:
            _resume['dirty'] = False
            ensure_movable()
        _movej(*a, **kw)
        mwait(time=WAIT_MOTION)
        check_estop_and_wait()

    r = SPEED_RATIO
    tr = TOSS_SPEED_RATIO

    def sc_toss(x):
        return [v * tr for v in x] if isinstance(x, (list, tuple)) else x * tr

    def apply_speed(ratio):
        """조리 속도 일괄 설정. 안전 정리 구간에서 저속으로 낮췄다가 원복하는 데 쓴다."""
        set_velj(60.0 * ratio)
        set_accj(100.0 * ratio)
        set_velx(250.0 * ratio, 80.625 * ratio)
        set_accx(1000.0 * ratio, 322.5 * ratio)

    # ── 그리퍼 DO 제어 ───────────────────────────────────────────────────

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

    def grip_neutral():
        """DO 전부 OFF — 아무 명령도 걸지 않은 중립 상태.

        ★ v5 신규. 사람이 DI14/DI15로 그리퍼를 만질 때 쓴다. DI14/15는
        estop_button_io_integrate가 Modbus로 폭을 직접 움직이는 명령이라(닫기 0mm /
        열기 110mm, 20N), 이쪽에서 DO로 반대 명령을 물고 있으면 서로 싸운다.
        (probe_flip3/v4도 웍 파지 직전에 set_digital_output(2, OFF)로 같은 중립화를 한다)"""
        set_digital_output(1, OFF)
        set_digital_output(2, OFF)
        set_digital_output(3, OFF)

    def resync_gripper_do_closed():
        """사람이 Modbus(DI14)로 만들어준 파지를 DO 상태에도 반영한다.

        ★ v5 신규. 이걸 안 하면 DO는 중립인데 실제 그리퍼만 닫혀 있는 불일치 상태로
        조리가 재개되고, 이후 코드가 DO를 만지는 순간 사람이 만든 파지가 풀린다."""
        set_digital_output(2, OFF)
        set_digital_output(3, OFF)
        set_digital_output(1, ON)

    def shake_ladle():
        ret = move_periodic(amp=[20.00, 0.00, 20.00, 0.00, 0.00, 0.00],
                            period=[0.50, 0.00, 0.50, 0.00, 0.00, 0.00],
                            atime=0.00, repeat=3, ref=0)
        if ret != 0:
            logger.warn("재료 투입 흔들기(move_periodic) 실패 — ret=%s" % ret)
        mwait(time=0.50)

    # ── 위치 확인 ────────────────────────────────────────────────────────

    def tcp_offset_mm(expected):
        """현재 TCP와 기대 위치의 거리(mm). 읽기 실패면 None."""
        try:
            pos, _sol = get_current_posx()
        except Exception:
            return None
        if pos is None:
            return None
        try:
            return math.dist([float(pos[i]) for i in range(3)],
                             [float(expected[i]) for i in range(3)])
        except Exception:
            return None

    def approach_wok_grip_safely():
        """P_WOK_GRIP으로 접근. 현재 위치가 기대와 크게 다르면 직선(movel)으로 직행하지 않고
        알려진 조인트 자세(J_WOK_APPROACH)를 먼저 거친다.

        ★ v5 신규. 사람이 개입한 뒤에는 로봇이 조그로 옮겨져 있을 수 있는데, 그 상태에서
        movel을 그대로 내보내면 예상과 전혀 다른 직선 경로가 나간다."""
        d = tcp_offset_mm(_P_WOK_GRIP)
        if d is None:
            logger.warn('현재 TCP를 읽지 못함 — 안전을 위해 J_WOK_APPROACH를 거쳐 접근합니다')
            movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        elif d > POSE_TOLERANCE_MM:
            logger.warn('현재 TCP가 웍 파지 위치에서 %.0fmm 벗어남 — J_WOK_APPROACH 경유' % d)
            movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        else:
            logger.info('현재 TCP가 웍 파지 위치 근처 (%.0fmm) — 직접 접근' % d)
        movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    # ── 안전 정리 (단계 재시작 전에 반드시 거치는 경로) ──────────────────

    def safe_release_and_home():
        """★ v5의 핵심 수정. 단계를 처음부터 다시 돌리기 전에, 그리퍼가 들고 있는 것을
        제자리에 내려놓고 알려진 자세로 빠져나온다.

        v4는 이 단계가 없어서, 파지 실패로 정지 → 사람이 웍을 물려줌 → 재개 →
        '레버열기+웍파지' 단계를 처음부터 재시작 → lever_open()이 웍을 든 채 레버로
        날아가 웍을 떨어뜨렸다. 재료 국자를 든 상태에서 재료투입 단계를 재시작하는
        경우도 같은 종류의 사고가 난다(국자를 든 채 국자 자리로 다시 내려감)."""
        if _recovering['active']:
            return
        _recovering['active'] = True
        item = _held['item']
        try:
            apply_speed(RECOVERY_SPEED_RATIO)
            logger.warn('안전 정리 시작 — 그리퍼 보유물=%s' % (item or '없음'))

            if item == 'wok':
                # 웍을 든 채로는 어디로도 못 간다. 화구 경유점을 거쳐 내려놓는다.
                logger.warn('웍을 들고 있음 — 화구(P_WOK_GRIP)에 내려놓고 개방합니다')
                wok_lower_to_stove()
                grip_open()
                wait(GRIP_WAIT)
            elif item == 'lever':
                # 레버 손잡이는 그 자리에서 놓고 물러난다.
                logger.warn('레버를 잡고 있음 — 그 자리에서 개방 후 물러납니다')
                grip_open()
                wait(GRIP_WAIT)
                movej(J_LEVER_OPEN_NEUT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
            elif item == 'ladle1':
                logger.warn('재료 국자1을 들고 있음 — 제자리에 되돌립니다')
                movel(P_ING1_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                movel(P_ING1_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                grip_open()
                wait(GRIP_WAIT)
                movel(P_ING1_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
            elif item == 'ladle2':
                logger.warn('재료 국자2를 들고 있음 — 제자리에 되돌립니다')
                movel(P_ING2_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                movel(P_ING2_RELEASE, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                grip_open()
                wait(GRIP_WAIT)
                movel(P_ING2_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
            elif item == 'ladle3':
                logger.warn('재료 국자3을 들고 있음 — 제자리에 되돌립니다')
                movel(P_ING3_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                movel(P_ING3_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
                grip_open()
                wait(GRIP_WAIT)
                movel(P_ING3_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
            elif item == 'scoop':
                # 주걱은 준비 단계에서 잡지 않으므로 여기로 오지 않는 게 정상.
                logger.error('주걱을 든 채 단계 재시작 요청 — 자동 정리 경로가 없습니다. '
                             '주걱을 사람이 회수한 뒤 재개하세요.')
                trigger_estop_and_wait('주걱 보유 상태에서 복구 불가', alarm='GRIP_WIDTH_ABNORMAL')
                _held['item'] = None
            else:
                grip_open()
                wait(GRIP_WAIT)

            _held['item'] = None
            publish_grip_state('NOT_GRIPPED')

            # 알려진 자세로 빠져나온다. 여기서부터는 어느 단계를 처음부터 돌려도 안전하다.
            movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
            movej(J_HOME, radius=0.00, ra=DR_MV_RA_DUPLICATE)
            logger.info('안전 정리 완료 — 홈 자세')
        finally:
            apply_speed(_speed['base'])
            _recovering['active'] = False

    # ── 파지 + 검증 ──────────────────────────────────────────────────────

    def grip_close_and_verify(label, target_mm, tolerance_mm, close_wait, hold_as):
        """그리퍼를 닫고 폭+파지감지를 검증한다. 성공 True / WOK_GRIP_WIDTH_RETRY회
        모두 실패하면 False. (v4의 grip_and_verify_width에서 '실패 시 비상정지'를 분리해
        호출부가 실패를 다룰 수 있게 했다 — v4는 여기서 바로 정지해버려서 웍질 중 실패가
        검증 없이 통과되는 경로가 생겼다.)"""
        for attempt in range(1, WOK_GRIP_WIDTH_RETRY + 1):
            set_digital_output(2, OFF)
            set_digital_output(1, ON)
            wait(close_wait)

            ok, width, detected = check_grip(target_mm, tolerance_mm)

            if ok is None:
                # 폭 읽기 실패. v4는 무제한으로 '검증 없이 진행'했지만, 그러면 Modbus가
                # 끊긴 동안 폭 검증이 통째로 무력화된 채 조리가 계속된다.
                _modbus_fail['count'] += 1
                if _modbus_fail['count'] >= MODBUS_FAIL_LIMIT:
                    logger.error('%s 파지 폭 읽기 %d회 연속 실패 — 검증 불능 상태로는 진행하지 '
                                 '않습니다 (Modbus 연결/동시접속 확인)'
                                 % (label, _modbus_fail['count']))
                    return False
                logger.warn('%s 파지 폭 읽기 실패(%d/%d) — 이번은 검증 없이 진행합니다'
                            % (label, _modbus_fail['count'], MODBUS_FAIL_LIMIT))
                _held['item'] = hold_as
                publish_grip_state('GRIPPED')
                return True

            _modbus_fail['count'] = 0

            if ok:
                logger.info('%s 파지 정상 (%s, %d/%d회차)'
                            % (label, describe_grip(width, detected),
                               attempt, WOK_GRIP_WIDTH_RETRY))
                _held['item'] = hold_as
                publish_grip_state('GRIPPED')
                return True

            logger.warn('%s 파지 비정상 (%s, 목표 %.1f±%.1fmm, %d/%d회차)'
                        % (label, describe_grip(width, detected),
                           target_mm, tolerance_mm, attempt, WOK_GRIP_WIDTH_RETRY))
            if attempt < WOK_GRIP_WIDTH_RETRY:
                set_digital_output(1, OFF)
                set_digital_output(2, ON)
                wait(GRIP_WAIT)

        _held['item'] = None
        publish_grip_state('GRIP_FAILED')
        return False

    def stop_for_human_regrip(label, target_mm, tolerance_mm, reason):
        """★ v5의 두 번째 핵심 수정 — 파지 실패로 정지하고 사람 개입을 기다린 뒤,
        '로봇을 움직이기 전에' 사람이 만들어준 파지를 직접 재검증한다.

        반환값:
          'HELD'  — 사람이 제대로 물려줬다. 호출부는 재파지도 단계 재시작도 하지 말고
                    그 자리에서 하던 작업을 이어가면 된다.
          'RETRY' — 여전히 비정상. 호출부가 로봇 파지를 다시 시도한다.

        v4는 여기서 무조건 RestartPreparation을 던져(준비 단계면) 단계를 처음부터 돌렸다.
        사람이 이미 웍을 물려준 상태에서 lever_open()부터 다시 도는 게 그 결과였다."""
        # 1) 사람이 그리퍼를 자유롭게 쓰게 열어주고, DO는 중립으로 — DI14/15(Modbus 직접
        #    제어)와 DO가 서로 다른 명령을 물고 싸우지 않도록.
        grip_open()
        wait(GRIP_WAIT)
        grip_neutral()
        _held['item'] = None

        logger.error('─' * 60)
        logger.error('%s' % reason)
        logger.error('사람 개입 절차:')
        logger.error('  1) DI15(열기) 버튼')
        logger.error('  2) 그리퍼에 %s을(를) 끼운다' % label)
        logger.error('  3) DI14(닫기) 버튼')
        logger.error('  4) UI "작업 재개" 또는 DI16(재개) 버튼')
        logger.error('  재개 후 로봇은 움직이기 전에 파지를 다시 확인합니다.')
        logger.error('─' * 60)

        trigger_estop_and_wait(reason)

        # 2) 재개 직후 — 로봇을 아직 움직이지 않은 상태에서 상태부터 푼다.
        ensure_movable()

        # 3) 사람이 만들어준 파지 재검증.
        ok, width, detected = check_grip(target_mm, tolerance_mm)
        if ok:
            logger.info('재개 후 %s 파지 재검증 통과 (%s) — 그 자리에서 작업을 이어갑니다'
                        % (label, describe_grip(width, detected)))
            resync_gripper_do_closed()
            _held['item'] = 'wok' if label == '웍' else 'scoop'
            _modbus_fail['count'] = 0
            publish_grip_state('GRIPPED')
            return 'HELD'

        if ok is None:
            logger.warn('재개 후 %s 파지 폭을 읽지 못함 — 로봇이 다시 파지를 시도합니다' % label)
        else:
            logger.warn('재개 후 %s 파지 재검증 실패 (%s) — 로봇이 다시 파지를 시도합니다'
                        % (label, describe_grip(width, detected)))
        return 'RETRY'

    def wok_pick():
        """웍(프라이팬) 파지. 성공하면 True.

        WOK_PICK_CYCLE_MAX 사이클(각 사이클 = 로봇 3회 재시도 + 사람 개입 대기)까지
        시도한다. 사람이 물려준 파지가 재검증을 통과하면 그 즉시 True로 빠져나온다
        (재파지도 단계 재시작도 하지 않는다 — v4가 여기서 lever_open()을 다시 돌았다).

        모두 소진하면 False. 호출부는 이 False를 반드시 확인해야 한다 — v4는 반환값이
        없어서 파지 확보 실패 상태로 1.25배속 웍질에 그대로 진입했다."""
        for cycle in range(1, WOK_PICK_CYCLE_MAX + 1):
            approach_wok_grip_safely()
            wait(0.50)
            if grip_close_and_verify('웍', WOK_GRIP_WIDTH_TARGET_MM,
                                     WOK_GRIP_WIDTH_TOLERANCE_MM,
                                     WOK_GRIP_WAIT, hold_as='wok'):
                return True

            outcome = stop_for_human_regrip(
                '웍', WOK_GRIP_WIDTH_TARGET_MM, WOK_GRIP_WIDTH_TOLERANCE_MM,
                '웍 파지 %d회 연속 실패 (사이클 %d/%d)'
                % (WOK_GRIP_WIDTH_RETRY, cycle, WOK_PICK_CYCLE_MAX))
            if outcome == 'HELD':
                return True

        logger.error('웍 파지를 %d사이클 동안 확보하지 못했습니다.' % WOK_PICK_CYCLE_MAX)
        return False

    def scoop_pick():
        """주걱 파지. 웍과 동일한 조건(56±2mm)·동일한 실패 처리. 성공하면 True.

        호출 전에 J_SCOOP_GRIP 자세로 이동해 두어야 한다. 재시도는 그리퍼 개폐만 하고
        자세는 바꾸지 않는다(주걱 자리는 고정이라 재접근이 필요 없다)."""
        for cycle in range(1, WOK_PICK_CYCLE_MAX + 1):
            set_digital_output(3, OFF)   # 국자용 출력 중립화 (공용 루틴은 DO1/DO2만 만진다)
            if grip_close_and_verify('주걱', SCOOP_GRIP_WIDTH_TARGET_MM,
                                     SCOOP_GRIP_WIDTH_TOLERANCE_MM,
                                     2.00, hold_as='scoop'):
                return True

            outcome = stop_for_human_regrip(
                '주걱', SCOOP_GRIP_WIDTH_TARGET_MM, SCOOP_GRIP_WIDTH_TOLERANCE_MM,
                '주걱 파지 %d회 연속 실패 (사이클 %d/%d)'
                % (WOK_GRIP_WIDTH_RETRY, cycle, WOK_PICK_CYCLE_MAX))
            if outcome == 'HELD':
                return True

            # 사람이 개입한 뒤에는 자세가 틀어져 있을 수 있으므로 파지 자세로 다시 잡는다.
            movej(J_SCOOP_GRIP, radius=0.00, ra=DR_MV_RA_DUPLICATE)

        logger.error('주걱 파지를 %d사이클 동안 확보하지 못했습니다.' % WOK_PICK_CYCLE_MAX)
        return False

    def wok_still_held():
        """웍질 직전 '아직 웍을 들고 있는지' 확인. grip_detected가 명시적으로 False일
        때만 실패로 본다(읽기 실패는 통과 — 여기서 조리를 멈출 만큼 확실한 신호가 아니다).

        ★ v5 신규. v4는 웍질 도중 웍을 놓쳐도 1.25배속 토스를 끝까지 돌았다."""
        detected = read_gripper_grip_detected()
        if detected is False:
            logger.error('웍질 직전 파지감지 상실 (grip_detected=False)')
            return False
        return True

    logger.info("wok_integrate5 시작 — dish=%s, SPEED_RATIO=%.2f, FLIP_SPEED_RATIO=%.2f, SIMULATION=%s"
                % (dish, r, FLIP_SPEED_RATIO, SIMULATION))

    set_singular_handling(DR_AVOID)

    # ── 볶음밥(fried_rice) 서브루틴 (wok_final3.py 기준) ───────────────────

    def ingredients1():
        movel(P_ING1_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING1_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_ladle()
        wait(GRIP_WAIT)
        _held['item'] = 'ladle1'
        movel(posx(632.87, -577.14, 373.27, 0.86, 93.62, 88.49),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_APPROACH, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_ROT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        shake_ladle()
        movel(P_POUR_EXIT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING1_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING1_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        _held['item'] = None
        movel(P_ING1_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def ingredients2():
        movel(posx(520.87, -533.60, 76.91, 1.00, 94.12, 89.16),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(620.73, -528.87, 70.86, 1.12, 94.10, 88.96),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_ladle()
        wait(GRIP_WAIT)
        _held['item'] = 'ladle2'
        movel(P_ING2_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_APPROACH, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_ROT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        shake_ladle()
        movel(P_POUR_EXIT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING2_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING2_RELEASE, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        _held['item'] = None
        movel(P_ING2_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def ingredients3():
        movel(P_ING3_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING3_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_ladle()
        wait(GRIP_WAIT)
        _held['item'] = 'ladle3'
        movel(P_ING3_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_APPROACH, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_ROT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        shake_ladle()
        movel(P_POUR_EXIT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING3_LIFT, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_ING3_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        _held['item'] = None
        movel(P_ING3_STANDBY, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def lever_open():
        movej(J_LEVER_OPEN_NEUT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(P_LEVER_OPEN_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_close()
        wait(GRIP_WAIT)
        _held['item'] = 'lever'
        movej(J_LEVER_OPEN_PULL, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        _held['item'] = None
        movej(J_LEVER_OPEN_NEUT, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def lever_close():
        movej(J_LEVER_CLOSE_ENTRY, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_LEVER_NEUTRAL, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(P_LEVER_CLOSE_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        grip_close()
        wait(GRIP_WAIT)
        _held['item'] = 'lever'
        movej(J_LEVER_CLOSE_PULL, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        grip_open()
        wait(GRIP_WAIT)
        _held['item'] = None
        movej(J_LEVER_NEUTRAL, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def scoop_stir():
        """힘제어 진입 후 실제 젓기 구간 (wok_final3 변경분, v4와 동일).

          - 목표력 -10N (팬 바닥까지 확실히 눌러 넣기)
          - 원샷 체크 대신 0.5초 간격 폴링(최대 약 10초). 힘이 목표 시점보다 늦게 오르거나
            순간적으로 스쳐 지나가면 원샷 체크는 그대로 놓친다.
          - 판정 상한 8 → 15
          - 젓기: move_spiral → move_periodic(ref=DR_TOOL). move_spiral은 이 코드베이스에서
            실동작이 검증된 적이 없다(회전 평면이 팬 표면에 막히면 움직임이 사라짐).
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
        """힘제어 기반 국자 삽입 (wok_final3.py 기준). 성공하면 True.

        ★ v5: 주걱 파지에 실패하면 False를 반환해 힘제어 구간에 진입하지 않는다.
        빈 그리퍼로 팬 바닥을 -10N으로 누르면 그리퍼/팬이 상한다."""
        movej(J_SCOOP_STANDBY, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_SCOOP_GRIP, radius=0.00, ra=DR_MV_RA_DUPLICATE)   # 주걱 잡기

        if not scoop_pick():
            logger.error('주걱 파지 실패 — scoop 구간을 건너뜁니다')
            movej(J_SCOOP_STANDBY, radius=0.00, ra=DR_MV_RA_DUPLICATE)
            return False

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
        movel(P_SCOOP_RELEASE, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(2, ON)
        wait(GRIP_WAIT)
        _held['item'] = None
        movej(J_SCOOP_STANDBY, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        return True

    def toss(count, label):
        """웍질 토스 반복. radius 블렌딩으로 이어지는 연속 동작이라 wait_motion 미적용
        (raw _movej 사용) — estop 체크포인트는 한 번의 웍질(toss) 완료 시점에만 둔다.

        ★ wok_final3 반영: 마지막 토스는 radius=0으로 완전히 멈춰서, 뒤이은
        movel(P_WOK_PLACE_VIA)가 블렌딩 관성 없이 진짜 직선으로 나가도록 한다."""
        for i in range(count):
            publish_stage("웍질 %s %d/%d" % (label, i + 1, count))
            last = (i == count - 1)
            _movej(J_TOSS_A, vel=sc_toss(75.62), acc=sc_toss(330.48),
                   radius=50.00, ra=DR_MV_RA_DUPLICATE)
            _movej(J_TOSS_B, vel=sc_toss(85.02), acc=sc_toss(408.33),
                   radius=(0.00 if last else 50.00), ra=DR_MV_RA_DUPLICATE)
            check_estop_and_wait()

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
        _held['item'] = None
        publish_grip_state('NOT_GRIPPED')

    def pour_to_bowl(lift_after=False):
        """웍을 들고 그릇에 붓기. wok_final3에서 볶음밥 붓기 좌표가 부침개와 같아져
        두 조리가 공용한다. 유일한 차이는 붓고 나서 위로 70mm 빼는 동작(부침개만)."""
        movel(P_POUR_LIFTOFF, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movej(J_POUR_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(P_POUR_BOWL, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        if lift_after:
            movel(posx(0.00, 0.00, 70.00, 0.00, 0.00, 0.00),
                  radius=0.00, ref=0, mod=DR_MV_MOD_REL, ra=DR_MV_RA_DUPLICATE)
        movej(J_POUR_RETRACT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_POUR_RETURN, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def wokking():
        """준비 단계(레버+웍파지)가 끝난 상태에서 시작. 1차 웍질→scoop→재파지→본 웍질→붓기→원위치.

        ★ v5: 웍질에 들어가기 전마다 파지를 확인한다. v4는 재파지 실패를 무시하고
        TOSS_SPEED_RATIO=1.25 고속 토스로 그대로 진입했다. 반환 False면 호출부가
        레버 닫기/홈 복귀만 하고 조리를 끝낸다."""
        if not wok_still_held():
            logger.error('1차 웍질 직전 웍 파지 확인 실패 — 웍질을 시작하지 않습니다')
            return False

        _movej(J_TOSS_A, vel=sc_toss(75.62), acc=sc_toss(330.48),
               radius=50.00, ra=DR_MV_RA_DUPLICATE)
        toss(WOK_INNER, "1차")

        wok_place()
        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)

        publish_stage("scoop 시작")
        scoop()
        publish_stage("scoop 완료")

        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        if not wok_pick():
            logger.error('scoop 후 웍 재파지 실패 — 본 웍질을 시작하지 않습니다')
            return False

        for set_i in range(WOK_OUTER):
            publish_stage("웍질 세트 %d/%d" % (set_i + 1, WOK_OUTER))
            if not wok_still_held():
                logger.error('웍질 세트 %d 직전 파지 상실 — 웍질을 중단합니다' % (set_i + 1))
                return False
            toss(WOK_INNER, "세트%d" % (set_i + 1))
            # 웍을 잡은 채 화구 원위치에서 휴지. 경유점 경로(wok_final3)로 내려간다.
            wok_lower_to_stove()
            wait(SET_REST_WAIT)

        pour_to_bowl(lift_after=False)
        wok_place()
        return True

    # ── 부침개(jeon) 서브루틴 (probe_flip3.py 기준) ────────────────────────

    def jeon_flip():
        """뒤집기 반복. 던지고(FLAP) 받는(FLAT) 사이는 원본(probe_flip3)에서 50ms로
        정교하게 튜닝된 타이밍이라, toss()와 동일하게 raw _movej/_movel 사용 + estop
        체크포인트는 한 뒤집기 사이클 완료 시점에 한 번만 둔다. 공통 movel/movej 래퍼를
        매 서브동작마다 통과시키면 이 타이밍이 깨져 던지고 받는 동작 자체가 어긋난다.

        ★ v5: 사이클 시작 전마다 파지를 확인한다 — 웍을 놓친 채 500deg/s로 던지는
        동작을 반복하지 않도록."""
        for i in range(FLIP_COUNT):
            if not wok_still_held():
                logger.error('뒤집기 %d/%d 직전 파지 상실 — 뒤집기를 중단합니다'
                             % (i + 1, FLIP_COUNT))
                return False
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
        return True

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

    # ★ v5: v4의 '레버열기+웍파지' 한 단계를 둘로 쪼갰다. 웍 파지에서 실패해 이 단계를
    #   재시작해도 레버 동작을 다시 끌고 오지 않는다 — v4가 웍을 든 채 lever_open()을
    #   다시 돌던 경로가 구조적으로 사라진다.
    def stage_lever_open():
        publish_stage("레버 열기")
        lever_open()

    def stage_wok_grip():
        publish_stage("웍 파지")
        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        if not wok_pick():
            logger.error('웍 파지 확보 실패 — 안전 정리 후 이 단계를 다시 시도합니다')
            safe_release_and_home()
            raise RestartPreparation()

    def stage_jeon_home_and_ingredients():
        publish_stage("홈 복귀")
        movej(J_HOME, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        set_velx(250.0 * INGREDIENT_SPEED_RATIO, 80.625 * INGREDIENT_SPEED_RATIO)
        set_accx(1000.0 * INGREDIENT_SPEED_RATIO, 322.5 * INGREDIENT_SPEED_RATIO)
        publish_stage("재료 투입(반죽)")
        ingredients2()
        set_velx(250.0 * FLIP_SPEED_RATIO, 80.625 * FLIP_SPEED_RATIO)
        set_accx(1000.0 * FLIP_SPEED_RATIO, 322.5 * FLIP_SPEED_RATIO)

    def stage_jeon_lever_open():
        movel(posx(544.14, -498.37, 292.14, 1.33, 93.74, 89.43),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        publish_stage("레버 열기")
        lever_open()
        set_digital_output(2, OFF)   # probe_flip3와 동일 — 웍 파지 전 출력 중립화

    PREP_STAGES_FRIED_RICE = [
        ('홈복귀+재료투입1', stage_home_and_ingredients1),
        ('재료투입2', stage_ingredients2),
        ('재료투입3', stage_ingredients3),
        ('레버열기', stage_lever_open),
        ('웍파지', stage_wok_grip),
    ]
    PREP_STAGES_JEON = [
        ('홈복귀+재료투입', stage_jeon_home_and_ingredients),
        ('레버열기', stage_jeon_lever_open),
        ('웍파지', stage_wok_grip),
    ]

    # ── 메인 시퀀스: dish 파라미터로 분기 ───────────────────────────────

    if dish == 'fried_rice':
        _speed['base'] = r
        apply_speed(r)

        run_preparation(PREP_STAGES_FRIED_RICE)

        publish_stage("웍질 시작")
        ok = wokking()

        if not ok:
            publish_stage("웍질 중단 — 안전 정리")
            safe_release_and_home()

        movel(P_POST_RELEASE, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

        publish_stage("레버 닫기")
        lever_close()

        movej(J_HOME, radius=100.00, ra=DR_MV_RA_DUPLICATE)

    else:  # jeon
        fr = FLIP_SPEED_RATIO
        _speed['base'] = fr
        apply_speed(fr)

        run_preparation(PREP_STAGES_JEON)

        publish_stage("뒤집기 시작")
        ok = jeon_flip()

        if ok:
            publish_stage("그릇에 붓기")
            pour_to_bowl(lift_after=True)
            publish_stage("웍 내려놓기")
            wok_place()
        else:
            publish_stage("뒤집기 중단 — 안전 정리")
            safe_release_and_home()

        movel(P_POST_RELEASE, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

        publish_stage("레버 닫기")
        lever_close()

        movej(J_HOME, radius=30.00, ra=DR_MV_RA_DUPLICATE)

    publish_stage("완료" if ok else "중단됨(안전 정리 후 종료)")
    gripper_client.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
