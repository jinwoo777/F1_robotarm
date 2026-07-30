import rclpy
from rclpy.logging import get_logger

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"

SPEED_RATIO = 0.6   # 홈/레버 이동 속도. 재료 픽업 속도와 통일 (wok_final.py 와 동일)
INGREDIENT_SPEED_RATIO = 0.6   # 재료 투입(ingredients2) 전용 속도 (wok_final.py 와 동일)

FLIP_COUNT = 3                 # 뒤집기 반복 횟수
FLIP_PAUSE_SEC = 1.0            # 던지고 나서 다음 던지기 전까지 대기 시간(초). 2.0 은 너무 길어서 낮춤.

VELOCITY_J = 10    # 초기 위치 접근 + down(내려가기) 전용 — 느려도 됨
ACC_J = 20
WOK_VEL_J = 500    # flap(위로 던지기) 전용 — 빠르게 (그대로 유지)
WOK_ACC_J = 1000   # 전이 360도만 도는 문제 -> 540도까지 가도록 아주 조금만 상향 (850 -> 950)

READY_JOINT = [3.01, 8.44, 53.86, -2.55, 72.76, 3.06]
FLAP_JOINT = [3.01, 5.0, 43.00, -2.55, 45, 3.06]
FLAT_JOINT = [3.01, 5.0, 43.00, -2.55, 50, 3.06]   # flap 과 J5 차이(7.5도)는 그대로 유지

import DR_init
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
logger = get_logger("probe_flip")

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("probe_flip", namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import(
        set_singular_handling, set_velj, set_accj, set_velx, set_accx,
        movej, movel, move_periodic, wait, mwait, set_digital_output,
        posj, posx,
        DR_AVOID, DR_MV_MOD_ABS, DR_MV_MOD_REL, DR_MV_RA_DUPLICATE,
        ON, OFF,
    )
    r = SPEED_RATIO

    set_singular_handling(DR_AVOID)
    set_velj(60.0 * r)
    set_accj(100.0 * r)
    set_velx(250.0 * r, 80.625 * r)
    set_accx(1000.0 * r, 322.5 * r)

    def lever_open():
        movej(posj(-21.56, 45.45, 31.34, -0.05, 99.96, -16.28),
            radius=0.00, ra=DR_MV_RA_DUPLICATE)
        movel(posx(633.19, -239.32, 41.26, 163.12, -176.86, 168.11),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(3, OFF)
        set_digital_output(2, OFF)
        set_digital_output(1, ON)
        wait(2.00)
        movej(posj(-21.50, 45.35, 49.67, -0.04, 81.85, 69.22),
            radius=0.00, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)
        wait(2.00)
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
        wait(2.00)
        movej(posj(-21.50, 45.35, 49.67, -0.04, 81.85, -23.42),
            radius=0.00, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)
        wait(2.00)
        movej(posj(-21.56, 45.45, 31.34, -0.05, 99.96, -16.28),
            radius=0.00, ra=DR_MV_RA_DUPLICATE)

    def shake_ladle():
        """재료 투입구에서 탈탈 흔들기 (wok_final.py 와 동일). 실패해도 예외를 안 내므로 ret 확인."""
        ret = move_periodic(amp=[20.00, 0.00, 20.00, 0.00, 0.00, 0.00],
                            period=[0.50, 0.00, 0.50, 0.00, 0.00, 0.00],
                            atime=0.00, repeat=3, ref=0)
        if ret != 0:
            logger.warn("재료 투입 흔들기(move_periodic) 실패 — ret=%s" % ret)
        wait(0.50)

    def ingredients2():
        """재료2 파지 → 투입구(재료 그릇)로 이동 → 붓기 → 원위치 (wok_final.py ingredients2 이식)."""
        movel(posx(520.87, -533.60, 76.91, 1.00, 94.12, 89.16),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(620.73, -528.87, 70.86, 1.12, 94.10, 88.96),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(2, OFF)
        set_digital_output(3, ON)
        wait(2.00)
        movel(posx(620.73, -528.87, 370.86, 1.12, 94.10, 88.96),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(903.68, -48.07, 291.30, 31.02, 98.82, 93.12),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(903.68, -48.06, 291.30, 31.02, 98.82, -59.87),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        shake_ladle()
        movel(posx(904.95, -165.04, 270.56, 24.62, 98.82, 93.11),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(620.73, -528.87, 370.86, 1.12, 94.10, 88.96),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        movel(posx(623.80, -528.80, 72.11, 1.49, 93.96, 89.23),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
        set_digital_output(1, OFF)
        set_digital_output(3, OFF)
        set_digital_output(2, ON)
        wait(2.00)
        movel(posx(532.01, -532.66, 79.35, 1.37, 94.09, 89.43),
            radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    def flip():
        ready_joint = posj(*READY_JOINT)
        flap_joint = posj(*FLAP_JOINT)
        flat_joint = posj(*FLAT_JOINT)
        # 처음 웍 잡을 때와 같은 자세 — 매 뒤집기 사이 화구에 내려놓는 위치로 재사용.
        stove_pos = posx(704.84, -21.97, 181.70, 178.01, -109.75, 178.85)

        for i in range(FLIP_COUNT):
            logger.info("뒤집기 %d/%d" % (i+1, FLIP_COUNT))

            # ① 들어올리기 (화구 -> READY)
            movej(
                ready_joint,
                vel=VELOCITY_J,
                acc=ACC_J,
                radius=0,
                ra=DR_MV_RA_DUPLICATE
            )
            wait(0.3)

            # ② 순간적으로 확 던지기
            movej(
                flap_joint,
                vel=WOK_VEL_J,
                acc=WOK_ACC_J,
                radius=0,
                ra=DR_MV_RA_DUPLICATE
            )
            wait(0.05)

            # ③ 팬을 받아주기
            movej(
                flat_joint,
                vel=WOK_VEL_J,
                acc=WOK_ACC_J,
                radius=0,
                ra=DR_MV_RA_DUPLICATE
            )
            wait(0.1)

            # ④ 화구에 내려놓기 (그리퍼는 계속 잡고 있는 상태)
            movel(
                stove_pos,
                radius=0.00,
                ref=0,
                mod=DR_MV_MOD_ABS,
                ra=DR_MV_RA_DUPLICATE
            )
            if i < FLIP_COUNT - 1:
                wait(3.00)

    logger.info("probe_flip 시작 — WOK_VEL_J=%d, WOK_ACC_J=%d, FLIP_COUNT=%d, FLIP_PAUSE_SEC=%.1f"
                % (WOK_VEL_J, WOK_ACC_J, FLIP_COUNT, FLIP_PAUSE_SEC))

    #==========================
    #play
    #==========================
    #home
    movej(posj(0.00, 0.00, 90.00, 0.00, 90.00, 0.00), 
        radius=0.00, ra=DR_MV_RA_DUPLICATE)

    logger.info("재료 투입(ingredients2) 시작")
    set_velx(250.0 * INGREDIENT_SPEED_RATIO, 80.625 * INGREDIENT_SPEED_RATIO)
    set_accx(1000.0 * INGREDIENT_SPEED_RATIO, 322.5 * INGREDIENT_SPEED_RATIO)
    ingredients2()
    set_velx(250.0 * r, 80.625 * r)
    set_accx(1000.0 * r, 322.5 * r)
    logger.info("재료 투입(ingredients2) 완료")

    movel(posx(544.14, -498.37, 292.14, 1.33, 93.74, 89.43),
        radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    logger.info("레버 열기")
    lever_open()
    set_digital_output(2, OFF)

    movej(posj(-4.10, -9.12, 133.07, -1.39, -17.77, -0.13),
        radius=0.00, ra=DR_MV_RA_DUPLICATE)
    wait(0.50)

    movel(posx(704.84, -21.97, 181.70, 178.01, -109.75, 178.85),
        radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
    wait(0.50)
    set_digital_output(2, OFF)
    set_digital_output(1, ON)
    wait(2.00)

    logger.info("뒤집기 시작")
    flip()

    # 그릇에 붓기 — wok_final.py wokking() 의 "4) 붓기" 섹션과 동일 좌표
    logger.info("그릇에 붓기")
    movel(posx(698.57, -14.97, 330.02, 178.04, -117.09, 178.50),
        radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
    movej(posj(11.32, -44.41, 118.37, 3.22, 36.75, -7.76),
        radius=0.00, ra=DR_MV_RA_DUPLICATE)
    # z 368.03 → 398.03 로 30mm 띄워서 그릇과 간격 확보 (wok_final.py 와 동일, 하드웨어 확인 필요)
    movel(posx(393.35, 208.74, 398.03, 24.66, 101.59, -137.26),
        radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
    # 붓고 나서 위로 살짝 빼기
    movel(posx(0.00, 0.00, 70.00, 0.00, 0.00, 0.00),
        radius=0.00, ref=0, mod=DR_MV_MOD_REL, ra=DR_MV_RA_DUPLICATE)
    movej(posj(11.56, -39.43, 124.18, 3.22, 25.94, -7.76),
        radius=0.00, ra=DR_MV_RA_DUPLICATE)
    movej(posj(-3.94, 3.44, 101.11, 3.22, 7.08, -7.76),
        radius=0.00, ra=DR_MV_RA_DUPLICATE)

    logger.info("웍 내려놓기")
    movel(posx(704.84, -21.97, 181.70, 178.01, -109.75, 178.85),
        radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
    set_digital_output(1, OFF)
    set_digital_output(2, ON)
    wait(2.00)
    movel(posx(611.20, -15.91, 214.31, 178.09, -111.38, 178.50),
        radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)

    logger.info("레버 닫기")
    lever_close()

    movej(posj(0.00, 0.00, 90.00, 0.00, 90.00, 0.00), radius=30.00, ra=DR_MV_RA_DUPLICATE)

    logger.info("probe_flip 완료")
    rclpy.shutdown()

if __name__ == "__main__":
    main()