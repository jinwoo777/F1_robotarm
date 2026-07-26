"""wok_rl_execute — project/robot/trajectory_exporter.py 가 만든 학습된 웍질 궤적(JSON)을
movesx()로 재생한다. wok_test4.py와 동일한 안전 관례(SPEED_RATIO 단계적 검증)를 따른다.

*** 이 스크립트는 이 세션에서 실행되지 않았다. 사용자가 로봇 옆에서 직접 실행할 것. ***

실행 전 체크리스트 (project/results/<run>/exported_trajectory_*.json 의 "note" 필드 및
feasibility_report.feasible 값도 함께 확인):
  1. `feasible: true` 인 궤적인지 확인 (project/robot/trajectory_validator.py 리포트).
  2. rx,ry,rz는 ZYZ 오일러 근사치 — 실제 posx 관례와 다를 수 있음. 저속에서 자세가
     이상하면 즉시 정지하고 tcp_pan_transform.py/trajectory_exporter.py의 오일러 변환부터 재검토.
  3. SPEED_RATIO=0.1~0.2 로 시작, 작업반경 비우고, 빈 팬(재료 없이)으로 1사이클만 먼저 실행.
  4. 문제 없으면 SPEED_RATIO를 단계적으로(0.3 -> 0.5 -> 1.0) 올리며 재확인.
  5. 재료는 가벼운 모형(쌀알 등) -> 목표 질량 순으로 단계적으로 투입.
  6. 비상정지 버튼에 항상 손을 댈 수 있는 위치에 있을 것.

  터미널: ros2 run rokey wok_rl_execute --json <exported_trajectory_path> --speed-ratio 0.1
"""
import argparse
import json

import rclpy
from rclpy.logging import get_logger

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
logger = get_logger("wok_rl_execute")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True,
                         help="project/robot/trajectory_exporter.py 가 생성한 exported_trajectory_*.json 경로")
    parser.add_argument("--speed-ratio", type=float, default=0.1,
                         help="안전을 위해 기본 매우 낮게(0.1). wok_test4.py 관례와 동일.")
    cli_args = parser.parse_args(args=args)

    with open(cli_args.json) as f:
        export = json.load(f)

    if not export["feasible"]:
        logger.error("이 궤적은 project/robot/trajectory_validator.py 검증에서 feasible=false 로 나왔습니다. "
                      "재학습/재검증 없이 실행하지 마세요.")
        return
    logger.warn(export["note"])

    rclpy.init(args=None)
    node = rclpy.create_node("wok_rl_execute", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movesx, posx, mwait, DR_MV_MOD_ABS

    r = cli_args.speed_ratio
    waypoints = [
        posx(wp["x"], wp["y"], wp["z"], wp["rx"], wp["ry"], wp["rz"])
        for wp in export["posx_waypoints_mm_deg"]
    ]

    logger.info(f"wok_rl_execute 시작 — mass={export['target_mass_g']}g, "
                f"SPEED_RATIO={r}, n_tosses={export['n_tosses']}")

    for toss_i in range(export["n_tosses"]):
        logger.info(f"웍질 {toss_i + 1}/{export['n_tosses']}")
        movesx(waypoints, vel=[300.0 * r, 80.0 * r], acc=[1000.0 * r, 300.0 * r],
               mod=DR_MV_MOD_ABS)
        mwait(time=0.3)

    logger.info("wok_rl_execute 완료")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
