"""wok_test7 — m0609_test7.drl 을 ROS2(DSR_ROBOT2)에서 실행.

프로세스: 재료 3종 투입(ingredients1~3) → 레버 열기 → 웍 파지 → 1차 웍질(WOK_INNER 회)
         → 웍 내려놓기 → scoop()(힘제어 국자 삽입) → 웍 재파지
         → 본 웍질(WOK_OUTER 세트 x WOK_INNER 회, 세트마다 5초 휴지)
         → 붓기 → 웍 원위치 → 레버 닫기 → 홈.

────────────────────────────────────────────────────────────────────────────
■ ROS2 API 비호환 인자 제거 — doosan-robot2(humble) dsr_common2/imp/DSR_ROBOT2.py 소스 기준 검증
────────────────────────────────────────────────────────────────────────────
  실제 시그니처:
    movej(pos, vel, acc, time, radius, mod, ra, v, a, t, r)              ← velx 인자 없음
    movel(pos, vel, acc, time, radius, ref, mod, ra, v, a, t, r)         ← app_type 인자 없음
    set_velx(vel1, vel2=DR_COND_NONE)                                    ← 2인자
    move_spiral(rev, rmax, lmax, vel, acc, time, axis, ref, v, a, t)     ← rad_dir/rot_dir/ra/radius 없음
    move_periodic(amp, period, atime=None, repeat=None, ref=DR_TOOL)     ← 원본과 동일

  따라서 제거한 인자:
    - movel(..., app_type=DR_MV_APP_NONE)  → app_type 제거
      (DR_MV_APP_NONE 상수 자체는 DSR_ROBOT2 에 존재하지만 movel 이 받지 않음)
    - set_velx(250.0, 80.625, DR_OFF)      → 3번째 인자 제거
    - movej(..., velx=1350.38)             → velx 제거 (웍질 토스 4곳)
    - move_spiral(..., rad_dir=DR_SPIRAL_OUTWARD, rot_dir=DR_ROT_FORWARD,
                       ra=DR_MV_RA_DUPLICATE, radius=0.00)  → 4개 모두 제거

  상수 존재 여부 확인 결과 (DSR_ROBOT2.py / DR_common2.py grep):
    존재  : ON, OFF, DR_AVOID, DR_MV_MOD_ABS, DR_MV_RA_DUPLICATE, DR_BASE, DR_TOOL,
            DR_AXIS_Z(=2), DR_FC_MOD_REL(=1), DR_COND_NONE
    없음  : DR_ON, DR_OFF          → set_digital_output 은 원본도 ON/OFF 사용, DR_OFF 는 제거된 인자였음
    없음  : DR_SPIRAL_OUTWARD, DR_ROT_FORWARD
            → 해당 인자가 API 에 아예 없으므로 로컬 상수도 불필요, 인자째 삭제
    힘제어 상수(DR_AXIS_Z/DR_BASE/DR_FC_MOD_REL)는 설치 버전 차이를 대비해 아래에서
    try/except 로 import 하고 없으면 로컬 상수로 대체함 (m0609_gear_force.py 와 동일 패턴).

────────────────────────────────────────────────────────────────────────────
■ ★★ check_force_condition 반환값 — DRL 과 의미가 반대. 그대로 옮기면 판정이 뒤집힘
────────────────────────────────────────────────────────────────────────────
    DRL  : if check_force_condition(...):        # 조건 충족 = True
    ROS2 : ret = 0 if (result.success == True) else -1
           → 조건 충족 = 0, 미충족 = -1
    파이썬에서 `if 0:` 은 False, `if -1:` 은 True 이므로 단순 이식 시 정반대로 동작함.
    이 파일은 `if check_force_condition(...) == 0:` 로 명시 비교함.
    (DSR_ROBOT2.py 의 check_force_condition() 본문 확인)

────────────────────────────────────────────────────────────────────────────
■ ★ 웍 파지 = 웍 내려놓기 통일 좌표 (원본 대비 의도적 수정)
────────────────────────────────────────────────────────────────────────────
  원본 test7.drl 은 웍을 놓는 좌표가 3종류로 흩어져 있었음:
    (1) 잡기            : posx(704.84, -21.97, 181.70, 178.01, -109.75, 178.85)  ← 기준
    (2) 놓기(세트 휴지) : posx(684.16, -31.52, 213.72, 176.98, -113.41, 178.67)
    (3) 놓기(scoop 전/최종) : posx(704.48, -23.78, 192.98, 177.69, -112.93, 178.48)
  → (2)(3) 을 전부 (1) = P_WOK_GRIP 으로 통일. "웍을 놓는 모든 movel = P_WOK_GRIP".
     서로 다른 위치에 웍을 두면 다음 사이클에서 파지 실패 위험이 있음.
     ※ (2) 의 684.16 좌표는 wok_test4 검증 때 화구 쪽으로 치우쳐 간섭 위험이 지적된 좌표이기도 함.
     (2026-07-25: 한때 (3)만 원본 좌표로 분리했었으나, 본 웍질 세트 휴지 위치(1)를
     기준으로 다시 전부 통일하기로 함.)

────────────────────────────────────────────────────────────────────────────
■ 좌표 상수화
────────────────────────────────────────────────────────────────────────────
  전체 파일을 훑어 2회 이상 반복되는 조인트/좌표는 모두 상단 상수(_J_* / _P_*)로 분리.
  0.01~0.04mm 만 차이나는 티칭 노이즈 수준의 중복 좌표는 하나로 통일함
  (M0609 반복정밀도 ±0.03mm — 이 차이는 측정 불가 수준). 통일 항목은 상수 옆에 원본 값 주석.
  posj/posx 래핑은 main() 안에서 수행 — DSR_ROBOT2 import 가 rclpy 노드 생성 이후여야 하기 때문.

────────────────────────────────────────────────────────────────────────────
■ 원본 대비 유의점
────────────────────────────────────────────────────────────────────────────
  - test7.drl 의 test() 서브루틴은 어디서도 호출되지 않아 이식하지 않음.
  - test7.drl 에는 붓기 자세에서 move_periodic(탈탈 흔들기)이 없음 (wok_test4.py 에는 있었음).
    한 번 추가했다가 하드웨어에서 오류가 나서 다시 뺌 — 붓기 구간은 흔들기 없이 그대로 유지.
  - 붓기 높이가 wok_test4 의 Z=372.31 → test7 은 Z=392.31 로 20mm 높아짐 (원본 그대로 유지).
  - lever_open 의 중립 자세가 lever_close 의 중립 자세와 다름 (원본 그대로 유지):
      lever_open  : J_LEVER_OPEN_NEUTRAL  = posj(-20.55, 43.31, 29.64, -0.39, 106.07, -23.54)
      lever_close : J_LEVER_NEUTRAL       = posj(-21.56, 45.45, 31.34, -0.05,  99.96, -16.28)
  - 그리퍼 DO 후 대기: 원본은 mwait(0.50) / mwait(1.00).
    DRL 은 프로그램 흐름이 모션보다 선행하므로 실효 대기가 더 길었을 수 있으나,
    이 포팅은 movel/movej 래퍼가 매 동작마다 모션 완료를 보장하므로 실효 대기 = 상수값 그대로.
    하드웨어에서 그리퍼가 덜 닫히면 GRIP_WAIT / WOK_GRIP_WAIT 를 올릴 것
    (하드웨어 검증된 wok_test4.py 는 3.0 초를 사용했음).

────────────────────────────────────────────────────────────────────────────
■ 안전
────────────────────────────────────────────────────────────────────────────
  - 기본값은 SPEED_RATIO=0.6(재료 픽업 속도와 통일). 처음 돌리는 구간이라면 일시적으로
    낮춰서(예: 0.3) WOK_OUTER=1, 웍 없이 경로부터 확인할 것.
  - TOSS_SPEED_RATIO 는 1.0(원본 속도 그대로)에서 시작. wok_test4 에서 동일 토스 자세를
    1.17 까지 올려 검증했으므로, 경로 확인 후 단계적으로 올릴 것.
  - 시뮬레이터 검증 시 SIMULATION=True — scoop() 의 힘제어 구간을 통째로 우회함.
    시뮬레이터에는 접촉 물리가 없어 set_desired_force(-5N, Z) 가 걸리면 TCP 가 계속 하강함.
    (check_force_condition 은 폴링 while 이 아니라 단발 if 라 무한루프 위험 자체는 없음)
  - 실행 전: 작업 반경 비우기, 비상정지에 손, 그리퍼/레버 DO 매핑 사전 확인
    (DO1=닫기, DO2=열기, DO3=국자/재료 파지 — 원본 DRL 사용 패턴 기준).
"""
import time

import rclpy
from rclpy.logging import get_logger

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# ▼▼▼ 안전 파라미터 ▼▼▼
SIMULATION = False    # True → scoop() 의 힘제어(compliance/force/spiral) 구간 우회. 시뮬레이터 검증용.
SPEED_RATIO = 0.6     # 전체 속도/가속 스케일(토스 제외 모든 movel/movej). 재료 집으러 갈 때 속도와 통일.
INGREDIENT_SPEED_RATIO = 0.6   # 재료 집으러 갈 때/투입할 때(ingredients1~3) 전용 속도.
TOSS_SPEED_RATIO = 1.25   # 웍질 토스(movej x2) 전용 배율. 원본 그대로가 1.0. wok_test4 는 1.17 까지 검증.
WOK_OUTER = 3          # 본 웍질 세트 수 (원본과 동일 3). 첫 테스트는 1 로 줄여도 됨
WOK_INNER = 7          # 세트당 웍질 반복 (원본은 5, +2). scoop 전 1차 웍질에도 동일 적용
WAIT_MOTION = 0.50     # 동작(movel/movej) 사이 wait_motion 시간. 웍질 토스 구간은 블렌딩 유지 위해 제외.
GRIP_WAIT = 1.50       # 재료/레버 그리퍼 개폐 대기 (원본 test7.drl: mwait 0.50)
WOK_GRIP_WAIT = 1.50   # 웍 파지/해제 대기 (원본 test7.drl: mwait 1.00)
SET_REST_WAIT = 5.00   # 웍질 세트 사이 휴지 (원본 wait 5.00)
# ▲▲▲

# ─────────────────────────────────────────────────────────────────────────
# 반복 좌표 상수 (숫자 원본). main() 안에서 posj()/posx() 로 래핑됨.
# ─────────────────────────────────────────────────────────────────────────

# --- 조인트 자세 ---
_J_HOME             = [0.00, 0.00, 90.00, 0.00, 90.00, 0.00]          # 시작/종료 홈 (2회)
_J_LEVER_NEUTRAL    = [-21.56, 45.45, 31.34, -0.05, 99.96, -16.28]    # 레버 닫기 중립 (2회)
_J_LEVER_OPEN_NEUT  = [-20.55, 43.31, 29.64, -0.39, 106.07, -23.54]   # 레버 열기 중립 (2회)
_J_WOK_APPROACH     = [-4.10, -9.12, 133.07, -1.39, -17.77, -0.13]    # 웍 파지 직전 경유 (3회)
_J_TOSS_A           = [-3.40, 8.62, 81.29, -0.28, 42.61, -0.05]       # 웍질 토스 자세 1
_J_TOSS_B           = [-3.34, -3.85, 81.09, -0.24, 55.05, -0.05]      # 웍질 토스 자세 2
_J_SCOOP_STANDBY    = [-39.77, 28.57, 55.08, -0.24, 96.06, 53.34]     # scoop 대기 (2회)
_J_SCOOP_TRANSIT    = [-47.32, 24.05, 36.21, -0.33, 119.71, 45.99]    # scoop 이송 경유 (2회)
_J_SCOOP_APPROACH   = [-4.11, 23.42, 36.90, 8.99, 67.66, 74.51]       # scoop 삽입 접근 (2회)

# --- 태스크 좌표 ---
# ★ 웍 파지 = 웍 내려놓기 통일 좌표. 원본의 684.16.../704.48... 을 전부 이 좌표로 대체함.
_P_WOK_GRIP         = [704.84, -21.97, 181.70, 178.01, -109.75, 178.85]
# P_WOK_GRIP 보다 높은(Z 286) 경유점. 웍을 내려놓을 때 직선으로 바로 내려가면 화로에
# 걸려서, 이 지점을 먼저 거친 뒤 P_WOK_GRIP 으로 내려가도록 함.
_P_WOK_PLACE_VIA    = [703.08, -21.33, 286.00, 177.81, -114.16, 179.21]

# 재료 투입 공용(ingredients1~3 에서 동일하게 사용)
_P_POUR_APPROACH    = [903.68, -48.07, 291.30, 31.02, 98.82, 93.12]   # 투입구 접근 (3회, 완전 동일)
_P_POUR_ROT         = [903.68, -48.06, 291.30, 31.02, 98.82, -59.87]  # 투입 회전 (3회)
#   └ 원본 ingredients1 은 (…, -48.05, 291.33, …) — Y 0.01 / Z 0.03 차이라 통일함
_P_POUR_EXIT        = [904.95, -165.04, 270.56, 24.62, 98.82, 93.11]  # 투입 후 이탈 (3회, 완전 동일)

# ingredients1
_P_ING1_STANDBY     = [517.78, -645.92, 80.11, 0.86, 94.19, 88.94]    # (2회, 완전 동일)
_P_ING1_GRIP        = [615.78, -642.28, 73.26, 0.73, 94.19, 88.74]    # (2회)
#   └ 원본 복귀 시 Z=73.25 — 0.01mm 차이라 통일함

# ingredients2
_P_ING2_LIFT        = [620.73, -528.87, 370.86, 1.12, 94.10, 88.96]   # (2회)
#   └ 원본 복귀 시 (620.74, -528.86, 370.82) — 최대 0.04mm 차이라 통일함

# ingredients3
_P_ING3_STANDBY     = [507.39, -433.03, 78.81, 1.59, 94.27, 89.62]    # (2회)
#   └ 원본 복귀 시 X=507.40 — 0.01mm 차이라 통일함
_P_ING3_GRIP        = [614.61, -423.47, 73.75, 1.43, 94.14, 89.23]    # (2회)
#   └ 원본 복귀 시 (614.61, -423.48, 73.78) — 최대 0.03mm 차이라 통일함
_P_ING3_LIFT        = [614.61, -423.47, 370.75, 1.43, 94.14, 89.23]   # (2회, 완전 동일)

# scoop
_P_SCOOP_TRANSIT_A  = [439.91, -356.18, 26.75, 136.39, -179.80, -130.78]  # (1회, 복귀 경로)
_P_SCOOP_TRANSIT_B  = [335.25, -356.98, 32.47, 168.59, 179.74, -98.19]    # (2회)

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
logger = get_logger("wok_test7")


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("wok_test7", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import (
        set_singular_handling, set_velj, set_accj, set_velx, set_accx,
        movej as _movej, movel as _movel, move_periodic,
        mwait, wait, set_digital_output,
        task_compliance_ctrl, release_compliance_ctrl, set_stiffnessx,
        set_desired_force, release_force, check_force_condition, get_tool_force,
        posj, posx,
        DR_AVOID, DR_MV_MOD_ABS, DR_MV_RA_DUPLICATE, DR_TOOL,
        ON, OFF,
    )

    # 힘제어 상수는 설치 버전에 따라 없을 수 있어 방어적으로 처리
    # (m0609_gear_force.py 에서 DR_ON/DR_OFF 를 로컬 상수로 대체했던 것과 동일 패턴)
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

    # posj / posx 래핑 -----------------------------------------------------
    J_HOME            = posj(_J_HOME)
    J_LEVER_NEUTRAL   = posj(_J_LEVER_NEUTRAL)
    J_LEVER_OPEN_NEUT = posj(_J_LEVER_OPEN_NEUT)
    J_WOK_APPROACH    = posj(_J_WOK_APPROACH)
    J_TOSS_A          = posj(_J_TOSS_A)
    J_TOSS_B          = posj(_J_TOSS_B)
    J_SCOOP_STANDBY   = posj(_J_SCOOP_STANDBY)
    J_SCOOP_TRANSIT   = posj(_J_SCOOP_TRANSIT)
    J_SCOOP_APPROACH  = posj(_J_SCOOP_APPROACH)

    P_WOK_GRIP        = posx(_P_WOK_GRIP)
    P_WOK_PLACE_VIA   = posx(_P_WOK_PLACE_VIA)
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

    # 래퍼 -----------------------------------------------------------------

    def movel(*args, **kwargs):
        _movel(*args, **kwargs)
        mwait(time=WAIT_MOTION)

    def movej(*args, **kwargs):
        _movej(*args, **kwargs)
        mwait(time=WAIT_MOTION)

    r = SPEED_RATIO
    tr = TOSS_SPEED_RATIO

    def sc_toss(x):
        """스칼라/리스트 모두 TOSS_SPEED_RATIO 로 스케일 (웍질 토스 전용)."""
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
        """재료 투입구에서 탈탈 흔들기. move_periodic 은 실패해도 예외를 안 내므로 ret 확인."""
        ret = move_periodic(amp=[20.00, 0.00, 20.00, 0.00, 0.00, 0.00],
                            period=[0.50, 0.00, 0.50, 0.00, 0.00, 0.00],
                            atime=0.00, repeat=3, ref=0)
        if ret != 0:
            logger.warn("재료 투입 흔들기(move_periodic) 실패 — ret=%s" % ret)
        mwait(time=0.50)

    logger.info("wok_test7 시작 — SPEED_RATIO=%.2f, TOSS=%.2f, WOK_OUTER=%d, WOK_INNER=%d, SIMULATION=%s"
                % (r, tr, WOK_OUTER, WOK_INNER, SIMULATION))

    set_singular_handling(DR_AVOID)
    set_velj(60.0 * r)
    set_accj(100.0 * r)
    set_velx(250.0 * r, 80.625 * r)
    set_accx(1000.0 * r, 322.5 * r)

    # ---- 서브루틴 ------------------------------------------------------

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

    def scoop():
        """원본 test7.drl 의 scoop() — 힘제어 기반 국자 삽입. 모션/로직 원본 유지.

        ROS2 이식에 따른 변경만 적용:
          - move_spiral 의 rad_dir / rot_dir / ra / radius 인자 제거 (API 에 없음)
          - check_force_condition 반환값 비교를 `== 0` 으로 명시 (0 = 조건 충족)
          - task_compliance_ctrl → set_desired_force 사이 time.sleep(1.0)
            (m0609_gear_force.py 에서 확인된 ROS2 타이밍 이슈 대응)
          - SIMULATION=True 이면 힘제어 구간 전체 우회
        """
        movej(J_SCOOP_STANDBY, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        # 주걱 잡기 (원본 -39.69, 32.44, 78.09, -0.24, 69.01, 53.35 에서 변경)
        movej(posj(-40.11, 35.35, 73.83, 0.66, 73.86, 51.94),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        grip_close()
        wait(2.00)
        # 잡고 빼는 위치
        movej(posj(-39.84, 31.64, 73.83, 0.36, 75.1, 51.94),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(P_SCOOP_TRANSIT_B, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movej(J_SCOOP_TRANSIT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_SCOOP_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(posj(-3.88, 32.47, 49.41, 4.47, 60.80, 74.52),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)

        # ---- 힘제어 구간 ----
        if SIMULATION:
            logger.warn("SIMULATION=True — scoop() 힘제어 구간 우회 (시뮬레이터엔 접촉 물리 없음)")
            wait(1.00)
        else:
            task_compliance_ctrl()
            set_stiffnessx([200.00, 200.00, 3000.00, 200.00, 200.00, 200.00], time=0.0)
            time.sleep(1.0)   # compliance 진입 완료 대기 (ROS2 타이밍 이슈 — 없으면 force 가 무시됨)
            set_desired_force([0.00, 0.00, -10.00, 0.00, 0.00, 0.00],
                              [0, 0, 1, 0, 0, 0], time=0.0, mod=_DR_FC_MOD_REL)
            wait(10.00)
            # ★ 원샷 체크(10초 대기 후 1회 확인) 대신 폴링으로 변경.
            # 힘이 목표 시점보다 늦게 오르거나 순간적으로 스쳐 지나가면 원샷 체크는 놓치기 쉬움
            # (m0609_gear_force.py 의 while True 폴링 패턴과 동일하게, 단 무한 대기는 위험하므로 타임아웃 추가).
            # ★ ROS2 는 0 = 조건 충족, -1 = 미충족. DRL 의 if 문과 의미가 반대이므로 == 0 비교.
            fc = -1
            for _ in range(20):   # 0.5초 간격 최대 20회 = 약 10초
                fc = check_force_condition(axis=_DR_AXIS_Z, min=2, max=15, ref=_DR_BASE)
                tf = get_tool_force(ref=_DR_BASE)
                logger.info("scoop 힘제어: Fz=%s N, fc=%s (0=충족)" % (tf[2] if tf != -1 else "?", fc))
                if fc == 0:
                    break
                wait(0.5)
            if fc == 0:
                wait(0.3)   # check_force_condition 직후 바로 이어 보내면 무시될 수 있어 여유를 둠
                # ★ move_spiral 은 이 코드베이스에서 한 번도 실제 동작이 검증된 적이 없어서
                # (axis=Z/ref=TOOL 이 실제 삽입 방향과 안 맞으면 회전 평면이 팬 표면에 막혀
                # 움직임이 사라질 수 있음), 이미 검증된 move_periodic 기반 흔들기로 대체함
                # (shake_ladle(), gear_force.py 의 힘제어 중 흔들기와 동일 패턴).
                sp_ret = move_periodic(amp=[25.00, 25.00, 0.00, 0.00, 0.00, 0.00],
                                        period=[0.60, 0.40, 0.00, 0.00, 0.00, 0.00],
                                        atime=0.00, repeat=15, ref=DR_TOOL)
                logger.info("scoop 젓기(move_periodic) 반환값 ret=%s (0=성공 기대)" % sp_ret)
            else:
                logger.warn("scoop 힘 조건 10초 내 미충족 — 젓기 스킵")
            release_force(time=0.0)
            release_compliance_ctrl()
        # ---- 힘제어 구간 끝 ----

        movej(J_SCOOP_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(J_SCOOP_TRANSIT, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(P_SCOOP_TRANSIT_B, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_SCOOP_TRANSIT_A, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(440.03, -355.20, 6.90, 133.30, -179.97, -133.95),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        # 원본은 여기서 DO3 를 건드리지 않으므로 grip_open() 헬퍼 대신 원본 그대로 2줄
        set_digital_output(1, OFF)
        set_digital_output(2, ON)
        wait(GRIP_WAIT)   # 원본엔 없음 — 개방 직후 바로 이동하면 국자를 끌 수 있어 추가
        movej(J_SCOOP_STANDBY, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def toss(count, label):
        """웍질 토스 반복. radius 블렌딩으로 이어지는 연속 동작이라
        wait_motion 미적용(raw _movej 사용). movej 의 velx= 인자는 ROS2 API 에 없어 제거.
        마지막 토스는 radius=0 으로 완전히 멈춰서, 뒤이은 movel(P_WOK_GRIP) 이 블렌딩
        관성 없이 진짜 직선으로 나가도록 함 (화로 간섭 방지)."""
        for i in range(count):
            logger.info("웍질 %s %d/%d" % (label, i + 1, count))
            last = (i == count - 1)
            _movej(J_TOSS_A, vel=sc_toss(75.62), acc=sc_toss(330.48),
                   radius=50.00, ra=DR_MV_RA_DUPLICATE)
            _movej(J_TOSS_B, vel=sc_toss(85.02), acc=sc_toss(408.33),
                   radius=(0.00 if last else 50.00), ra=DR_MV_RA_DUPLICATE)

    def wok_pick():
        """웍 파지 — 반드시 P_WOK_GRIP 에서."""
        movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(2, OFF)
        set_digital_output(1, ON)
        wait(WOK_GRIP_WAIT)

    def wok_place():
        """웍 내려놓기 — 파지 좌표와 동일한 P_WOK_GRIP 으로 통일(원본은 3종류였음).
        화로 간섭 방지를 위해 P_WOK_PLACE_VIA 를 먼저 거친 뒤 내려감."""
        movel(P_WOK_PLACE_VIA, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(2, ON)
        wait(GRIP_WAIT)

    def wokking():
        # 1) 웍 파지 → 1차 웍질
        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        wait(0.50)
        set_digital_output(2, OFF)
        set_digital_output(1, ON)
        wait(WOK_GRIP_WAIT)

        # 원본에 있던 토스 진입 movej 1회 (루프 밖)
        _movej(J_TOSS_A, vel=sc_toss(75.62), acc=sc_toss(330.48),
               radius=50.00, ra=DR_MV_RA_DUPLICATE)
        toss(WOK_INNER, "1차")

        # 2) 웍 내려놓고 scoop (원본 좌표 684.16... → P_WOK_GRIP 로 통일)
        wok_place()
        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)

        logger.info("scoop 시작")
        scoop()
        logger.info("scoop 완료")

        # 3) 웍 재파지 → 본 웍질 (WOK_OUTER 세트)
        movej(J_WOK_APPROACH, radius=0.00, ra=DR_MV_RA_DUPLICATE)
        wok_pick()

        for set_i in range(WOK_OUTER):
            logger.info("웍질 세트 %d/%d" % (set_i + 1, WOK_OUTER))
            toss(WOK_INNER, "세트%d" % (set_i + 1))
            # 화로 간섭 방지를 위해 P_WOK_PLACE_VIA 를 먼저 거친 뒤 내려감.
            movel(P_WOK_PLACE_VIA, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
            # 원본 좌표 684.16... → P_WOK_GRIP 로 통일 (웍을 잡은 채 원위치에서 휴지)
            movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
            wait(SET_REST_WAIT)

        # 4) 붓기
        movel(posx(698.57, -14.97, 330.02, 178.04, -117.09, 178.50),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movej(posj(11.32, -44.41, 118.37, 3.22, 36.75, -7.76),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        # 붓기 자세. 원본 test7.drl 에는 흔들기(move_periodic)가 없음.
        # z 368.03 → 398.03 로 30mm 띄워서 그릇과 간격 확보 (하드웨어 확인 필요).
        movel(posx(393.35, 208.74, 398.03, 24.66, 101.59, -137.26),
              radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movej(posj(11.56, -39.43, 124.18, 3.22, 25.94, -7.76),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movej(posj(-3.94, 3.44, 101.11, 3.22, 7.08, -7.76),
              radius=0.00, ra=DR_MV_RA_DUPLICATE)

        # 5) 웍 원위치 (원본 좌표 704.48... → P_WOK_GRIP 로 통일)
        wok_place()

    # ---- 메인 시퀀스 -----------------------------------------------------

    movej(J_HOME, radius=0.00, ra=DR_MV_RA_DUPLICATE)

    set_velx(250.0 * INGREDIENT_SPEED_RATIO, 80.625 * INGREDIENT_SPEED_RATIO)
    set_accx(1000.0 * INGREDIENT_SPEED_RATIO, 322.5 * INGREDIENT_SPEED_RATIO)
    logger.info("재료 투입 1/3")
    ingredients1()
    logger.info("재료 투입 2/3")
    ingredients2()
    logger.info("재료 투입 3/3")
    ingredients3()
    set_velx(250.0 * r, 80.625 * r)
    set_accx(1000.0 * r, 322.5 * r)
    wait(1.00)

    logger.info("레버 열기")
    lever_open()
    set_digital_output(2, OFF)

    logger.info("웍질 시작")
    wokking()

    movel(posx(611.20, -15.91, 214.31, 178.09, -111.38, 178.50),
          radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    logger.info("레버 닫기")
    lever_close()

    movej(J_HOME, radius=100.00, ra=DR_MV_RA_DUPLICATE)

    logger.info("wok_test7 완료")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    