"""wok_rl_fried_rice_execute — Reinforce의 학습된 fried_rice 정책 궤적
(trajectories_20fps/*.npz)을 Reinforce/scripts/build_real_robot_plan.py로 변환한
JSON plan을 읽어 movesx()로 재생한다.

*** 이 스크립트는 이 세션에서 실로봇에 실행되지 않았다. 사용자가 로봇 옆에서
    직접 실행할 것. ***

전제 조건 — 반드시 확인:
  1. 웍을 이미 P_WOK_GRIP(wok_integrate.py와 동일 좌표)에서 파지한 상태여야 한다.
     이 스크립트는 재료투입/레버/파지를 하지 않는다 — 순수 팬 궤적 재생만 한다.
  2. plan JSON의 motion_contract_scaled.within_caps가 true인지 이 스크립트가
     시작 시 자동으로 확인하고, false면 실행을 거부한다.
  3. build_real_robot_plan.py의 좌표 변환은 "P_WOK_GRIP 티칭 자세 = Reinforce
     시뮬레이션 t=0 pan pose"라는 공학적 가정에 기반한다 — URDF/T_base_tcp_teach
     기반 재검증이 아니다. 저속·무재료로 먼저 방향과 경로를 육안 확인할 것.
  4. --speed-ratio는 plan에 이미 반영된 안전 배속(약 6배 느림) 위에 추가로 곱해지는
     배율이다. 기본 0.1 그대로 최초 실행하고, 문제 없으면 0.3 -> 0.5 -> 1.0 순으로
     단계적으로 올릴 것. 1.0을 넘기지 말 것(그러면 plan의 cap 검증이 무의미해진다).
  5. 비상정지 버튼에 항상 손을 댈 수 있는 위치에 있을 것. 이 스크립트는 독립
     실행이며 wok_integrate.py의 E-STOP 토픽 인프라에 연결돼 있지 않다 — 실제
     안전정지는 wok_exception_handling의 estop_button_io_integrate(DI13)에 의존한다.

  터미널: ros2 run rokey wok_rl_fried_rice_execute \
            --plan /home/rokey/wok_wark/Reinforce/trajectories_20fps/060g_ep049_real_robot_plan.json \
            --speed-ratio 0.1
"""
import argparse
import json
import time

import rclpy
from rclpy.logging import get_logger

from rokey.onrobot_rg2 import RG2Client, RG2Error, UnitId

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

GRIPPER_MODBUS_HOST = "192.168.1.1"
GRIPPER_MODBUS_UNIT_ID = UnitId.QUICK_CHANGER

# movesx 단일 명령 waypoint 한도(Doosan 사양). plan은 이보다 촘촘하게(보통 220개)
# 저장돼 있으므로, 이 한도 안으로 등간격 서브샘플링한다.
MOVESX_MAX_WAYPOINTS = 100

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
logger = get_logger("wok_rl_fried_rice_execute")


def _stride_waypoints(waypoints: list[dict], max_count: int) -> list[dict]:
    if len(waypoints) <= max_count:
        return waypoints
    stride = -(-len(waypoints) // max_count)  # ceil division
    strided = waypoints[::stride]
    if strided[-1] is not waypoints[-1]:
        strided.append(waypoints[-1])  # 마지막 웨이포인트(원위치)는 항상 포함
    return strided


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", type=str, required=True,
        help="build_real_robot_plan.py가 생성한 *_real_robot_plan.json 경로",
    )
    parser.add_argument(
        "--speed-ratio", type=float, default=0.1,
        help="plan에 이미 반영된 안전 배속 위에 추가로 곱하는 배율. 1.0을 넘기지 말 것.",
    )
    parser.add_argument(
        "--skip-grip-check", action="store_true",
        help="그리퍼 Modbus 연결 없이도 강제 실행(비권장 — 웍을 실제로 잡고 있는지 확인 못함).",
    )
    cli_args = parser.parse_args(args=args)

    with open(cli_args.plan) as f:
        plan = json.load(f)

    contract = plan["motion_contract_scaled"]
    if not contract["within_caps"]:
        logger.error(
            "이 plan은 motion_contract_scaled.within_caps=false 입니다. "
            "build_real_robot_plan.py의 --time-scale-margin을 늘려 재생성한 뒤 사용하세요."
        )
        return
    if not (0.0 < cli_args.speed_ratio <= 1.0):
        logger.error("--speed-ratio는 0보다 크고 1.0 이하여야 합니다.")
        return

    for note in plan["safety"]["notes"]:
        logger.warn(f"[안전 참고] {note}")
    logger.warn(
        f"source: {plan['source_condition']} / "
        f"time_scale_applied={plan['time_scale_applied']:.2f} / "
        f"duration_s={plan['duration_s']:.1f}s / speed_ratio={cli_args.speed_ratio}"
    )

    if not cli_args.skip_grip_check:
        gripper_client = RG2Client(
            host=GRIPPER_MODBUS_HOST, unit_id=GRIPPER_MODBUS_UNIT_ID, timeout=1.0
        )
        try:
            gripper_client.connect()
            status = gripper_client.get_status()
            if not status.grip_detected:
                logger.error(
                    "그리퍼 grip_detected=False — 웍을 먼저 P_WOK_GRIP에서 파지한 뒤 "
                    "다시 실행하세요. (--skip-grip-check로 강제 실행 가능하지만 비권장)"
                )
                gripper_client.close()
                return
            logger.info("그리퍼 파지 확인됨(grip_detected=True) — 진행합니다.")
            gripper_client.close()
        except RG2Error as e:
            logger.error(
                f"그리퍼 Modbus 연결/조회 실패: {e} — 웍 파지 여부를 확인할 수 없어 "
                "중단합니다. (--skip-grip-check로 강제 실행 가능하지만 비권장)"
            )
            return

    rclpy.init(args=None)
    node = rclpy.create_node("wok_rl_fried_rice_execute", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import set_singular_handling, movesx, posx, wait, DR_AVOID, DR_MV_MOD_ABS

    # DSR_ROBOT2는 import 시점에 서비스 클라이언트를 새로 만든다. 방금 생성된 클라이언트가
    # dsr_controller2와 DDS로 아직 매칭되기 전에 첫 호출이 나가면 기본 QoS(RELIABLE+VOLATILE)
    # 특성상 그 요청이 재전송 없이 유실돼 응답을 영원히 못 받는다 — wok_integrate.py에서
    # 실제로 겪은 문제와 동일 원인이라 동일하게 대응한다.
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(1.0)

    set_singular_handling(DR_AVOID)

    r = cli_args.speed_ratio
    vel = [plan["vel_mm_s_deg_s"][0] * r, plan["vel_mm_s_deg_s"][1] * r]
    acc = [plan["acc_mm_s2_deg_s2"][0] * r, plan["acc_mm_s2_deg_s2"][1] * r]

    waypoints_full = plan["waypoints"]
    waypoints = _stride_waypoints(waypoints_full, MOVESX_MAX_WAYPOINTS)
    pos_list = [posx(*wp["posx_mm_deg"]) for wp in waypoints]

    logger.info(
        f"wok_rl_fried_rice_execute 시작 — waypoints {len(waypoints_full)} -> "
        f"{len(pos_list)}개로 서브샘플링, vel={vel}, acc={acc}"
    )

    movesx(pos_list, vel=vel, acc=acc, mod=DR_MV_MOD_ABS)
    wait(0.3)

    logger.info("wok_rl_fried_rice_execute 완료")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
