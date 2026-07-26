import time

import rclpy
import DR_init

# ───────────────────── Robot Config ─────────────────────
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

# 원본 DRL 설정: set_velj(60), set_accj(100), set_velx(250, ...), set_accx(1000, ...)
VELOCITY_J, ACC_J = 60, 100     # joint motion  (deg/s, deg/s^2)
VELOCITY_L, ACC_L = 250, 1000   # linear motion (mm/s, mm/s^2)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("rokey_gear_force", namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    try:
        # create_node 전에 작성하면 오류 남
        from DSR_ROBOT2 import (
            set_tool,
            set_tcp,
            movej,
            movel,
            wait,
            set_digital_output,
            set_singular_handling,
            task_compliance_ctrl,
            release_compliance_ctrl,
            set_desired_force,
            release_force,
            check_force_condition,
            move_periodic,
            ON,
            OFF,
            DR_BASE,
            DR_AXIS_Z,
            DR_FC_MOD_REL,
            DR_AVOID,
        )

        from DR_common2 import posx, posj

    except ImportError as e:
        node.get_logger().info(f"Error importing DSR_ROBOT2 : {e}")
        return

    # 반환값 0=성공, -1=실패(이름 미등록 등) — 실패해도 프로그램이 그냥 진행되므로 로그로 확인
    ret = set_tool("Tool Weight_1")
    node.get_logger().info(f"set_tool ret={ret}")
    ret = set_tcp("GripperDA_v1")
    node.get_logger().info(f"set_tcp ret={ret}")

    # ───────────────────── Positions ─────────────────────
    # 원본 DRL의 System_grip_pos3(펜던트 전역 변수)는 홈 자세로 대체함
    homej = posj([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])

    # planet gear 1 : pick 접근 → pick → 상승 → place 접근 → place
    p1_pick_app  = posx([303.53, -112.28, 45.61, 106.03, -177.52, 73.42])
    p1_pick      = posx([299.51, -107.18, 39.94, 112.96, -177.47, 80.24])
    p1_lift      = posx([299.50, -107.17, 174.34, 112.94, -177.47, 80.21])
    p1_place_app = posx([448.03, 41.37, 200.83, 1.36, 179.78, -31.37])
    p1_place     = posx([449.83, 41.47, 37.24, 165.21, 179.95, 132.48])

    # planet gear 2
    p2_pick_app  = posx([403.88, -99.67, 146.73, 57.99, -178.06, 24.32])
    p2_pick      = posx([403.90, -99.68, 40.42, 58.08, -178.06, 24.42])
    p2_lift      = posx([403.90, -99.68, 146.75, 58.08, -178.06, 24.42])
    p2_place_app = posx([550.14, 48.50, 206.45, 17.65, 179.77, -14.99])
    p2_place     = posx([553.64, 48.83, 37.66, 21.35, 179.76, -11.29])

    # planet gear 3
    p3_pick_app  = posx([346.60, -15.71, 178.49, 142.10, -179.67, 109.76])
    p3_pick      = posx([346.60, -15.71, 37.67, 142.12, -179.67, 109.77])
    p3_lift      = posx([346.61, -15.71, 178.50, 142.21, -179.67, 109.87])
    p3_place_app = posx([497.39, 132.79, 173.29, 37.19, -179.70, 4.79])
    p3_place     = posx([497.40, 132.79, 38.20, 37.38, -179.70, 4.98])

    # sun gear (마지막 삽입은 힘 제어로 수행)
    sun_pick_app  = posx([349.34, -74.19, 138.97, 38.14, -179.68, 6.18])
    sun_pick      = posx([349.36, -74.19, 41.68, 38.48, -179.68, 6.52])
    sun_lift      = posx([349.35, -74.19, 138.98, 38.35, -179.68, 6.39])
    sun_place_app = posx([498.86, 74.51, 125.72, 12.70, -179.69, -19.40])

    # ───────────────────── Gripper ─────────────────────
    def gripper_release():
        # 그리퍼 열기 (DO1 OFF → DO2 ON)
        set_digital_output(1, OFF)
        set_digital_output(2, ON)

    def gripper_grip():
        # 그리퍼 닫기 (DO2 OFF → DO1 ON)
        set_digital_output(2, OFF)
        set_digital_output(1, ON)

    # ───────────────────── Sub Routines ─────────────────────
    def sub_planet(pick_app, pick, lift, place_app, place):
        """유성기어 1개: pick 후 place (원본 sub_planet1~3 공통 시퀀스)"""
        gripper_release()
        movel(pick_app, vel=VELOCITY_L, acc=ACC_L)
        movel(pick, vel=VELOCITY_L, acc=ACC_L)
        gripper_grip()
        wait(1.0)
        movel(lift, vel=VELOCITY_L, acc=ACC_L)
        movel(place_app, vel=VELOCITY_L, acc=ACC_L)
        movel(place, vel=VELOCITY_L, acc=ACC_L)
        gripper_release()

    def sub_sun():
        """태양기어: pick 후 힘 제어로 삽입 (원본 sub_sun)"""
        gripper_release()
        movel(sun_pick_app, vel=VELOCITY_L, acc=ACC_L)
        movel(sun_pick, vel=VELOCITY_L, acc=ACC_L)
        gripper_grip()
        wait(1.0)
        movel(sun_lift, vel=VELOCITY_L, acc=ACC_L)
        movel(sun_place_app, vel=VELOCITY_L, acc=ACC_L)

        # 순응 제어 시작 (x, y는 유연하게 / z는 강성 유지)
        ret = task_compliance_ctrl(stx=[200.0, 200.0, 3000.0, 200.0, 200.0, 200.0], time=0.0)
        node.get_logger().info(f"task_compliance_ctrl ret={ret}")

        # ★ 순응 제어가 컨트롤러에서 완전히 활성화될 때까지 대기 (필수)
        #   이 대기 없이 set_desired_force를 보내면 명령이 무시되어 로봇이 내려가지 않음
        time.sleep(0.5)

        # -z 방향으로 20N 힘 인가 (ret=-1이면 명령 거부된 것)
        ret = set_desired_force(
            fd=[0.0, 0.0, -20.0, 0.0, 0.0, 0.0],
            dir=[0, 0, 1, 0, 0, 0],
            time=0.0,
            mod=DR_FC_MOD_REL,
        )
        node.get_logger().info(f"set_desired_force ret={ret}")

        # 외력이 10~15N에 도달하면 rz ±15도 주기 운동으로 기어를 맞물림
        # (ROS2의 check_force_condition은 조건 만족 시 0, 불만족 시 -1 반환)
        # ※ 스스로 내려가는데 10~15N 구간을 지나쳐 계속 파고들기만 하면
        #    max=15를 지우고 min=10만 남길 것 (10N 이상이면 바로 트리거)
        while True:
            if check_force_condition(axis=DR_AXIS_Z, min=10, max=15, ref=DR_BASE) == 0:
                move_periodic(
                    amp=[0.0, 0.0, 0.0, 0.0, 0.0, 15.0],
                    period=[0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
                    atime=0.0,
                    repeat=1,
                    ref=DR_BASE,
                )
                break

        # 힘 제어 / 순응 제어 해제
        release_force(time=0.0)
        release_compliance_ctrl()
        gripper_release()

    # ───────────────────── Main Sequence ─────────────────────
    set_singular_handling(DR_AVOID)

    movej(homej, vel=VELOCITY_J, acc=ACC_J)

    try:
        node.get_logger().info("Assembling planet gear 1")
        sub_planet(p1_pick_app, p1_pick, p1_lift, p1_place_app, p1_place)

        node.get_logger().info("Assembling planet gear 2")
        sub_planet(p2_pick_app, p2_pick, p2_lift, p2_place_app, p2_place)

        node.get_logger().info("Assembling planet gear 3")
        sub_planet(p3_pick_app, p3_pick, p3_lift, p3_place_app, p3_place)

        node.get_logger().info("Assembling sun gear (force control)")
        sub_sun()

        # 조립 지점에서 수직 상승 (원본의 movel(System_grip_pos3) 대체)
        movel(sun_place_app, vel=VELOCITY_L, acc=ACC_L)

        node.get_logger().info("Gear assembly finished")

    except KeyboardInterrupt:
        node.get_logger().info("Program Stopped")

    except Exception as e:
        node.get_logger().info(f"Robot Error: {e}")

    finally:
        # 힘/순응 제어가 켜진 상태로 종료되는 것 방지
        try:
            release_force(time=0.0)
            release_compliance_ctrl()
        except Exception:
            pass
        movej(homej, vel=VELOCITY_J, acc=ACC_J)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
