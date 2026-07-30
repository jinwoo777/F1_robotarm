"""웍(프라이팬) 파지 폭 검증 예외처리 — wok_integrate2.py에서 발췌.

실제 하드웨어에서 웍을 잡을 때 테두리를 잘못(헛) 잡는 경우가 잦아서 추가한 검증 로직이다.
정상 파지 시 그리퍼 폭은 56mm 근방이어야 하는데, 이 범위를 벗어나면 열었다 닫았다를
최대 3회 재시도하고, 그래도 범위 밖이면 물리 비상정지 버튼을 누른 것과 동일하게 처리한다.

주의: 아래 함수들은 원본(wok_integrate2.py)에서는 main() 함수 안에 정의된 중첩 함수(클로저)라
      logger / node / _estop / alarm_pub / estop_pub / read_gripper_width_mm /
      publish_grip_result / RestartPreparation / 로봇 이동 함수(movel 등)를 바깥 스코프에서
      그대로 끌어다 쓴다. 이 파일은 그 부분만 리뷰용으로 그대로 옮긴 것이라 단독 실행은 안 되고,
      전체 흐름 참고용이다.
"""

# ▼▼▼ 웍(프라이팬) 파지 폭 검증 — 실제 파지 실패가 잦아 추가한 예외 처리 ▼▼▼
# 정상 파지 시 그리퍼 폭은 56mm 근방이어야 한다. 이 범위를 벗어나면 웍 테두리를 잘못
# 물었거나 헛잡은 것으로 보고, 열었다 닫았다를 재시도한다.
WOK_GRIP_WIDTH_TARGET_MM = 56.0
WOK_GRIP_WIDTH_TOLERANCE_MM = 2.0
WOK_GRIP_WIDTH_RETRY = 3
# ▲▲▲


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


def wok_pick():
    """웍(프라이팬) 파지 — 반드시 P_WOK_GRIP에서. 그리퍼 Modbus 파지검증(wok_test4 이식)에
    더해, 파지 폭이 WOK_GRIP_WIDTH_TARGET_MM±WOK_GRIP_WIDTH_TOLERANCE_MM(56±2mm) 범위인지
    검증한다 — 실제로 웍을 잘못(테두리가 아니라 헛)잡는 경우가 잦아서 추가한 예외처리.
    범위를 벗어나면 열었다 닫았다를 최대 WOK_GRIP_WIDTH_RETRY회 재시도하고, 그래도 범위
    밖이면 비상정지(물리 버튼과 동일)로 넘겨 사람이 직접 확인하게 한다."""
    movel(P_WOK_GRIP, radius=0.00, ref=0, mod=DR_MV_MOD_ABS, ra=DR_MV_RA_DUPLICATE)
    wait(0.50)

    width = None
    for attempt in range(1, WOK_GRIP_WIDTH_RETRY + 1):
        set_digital_output(2, OFF)
        set_digital_output(1, ON)
        wait(WOK_GRIP_WAIT)

        width = read_gripper_width_mm()
        if width is None:
            logger.warn('웍 파지 폭 읽기 실패(Modbus 연결 확인) — 폭 검증 없이 진행합니다')
            publish_grip_result()
            return

        if abs(width - WOK_GRIP_WIDTH_TARGET_MM) <= WOK_GRIP_WIDTH_TOLERANCE_MM:
            logger.info('웍 파지 폭 정상 (%.1fmm, %d/%d회차)'
                        % (width, attempt, WOK_GRIP_WIDTH_RETRY))
            publish_grip_result()
            return

        logger.warn('웍 파지 폭 비정상 (%.1fmm, 목표 %.1f±%.1fmm, %d/%d회차)'
                    % (width, WOK_GRIP_WIDTH_TARGET_MM, WOK_GRIP_WIDTH_TOLERANCE_MM,
                       attempt, WOK_GRIP_WIDTH_RETRY))
        if attempt < WOK_GRIP_WIDTH_RETRY:
            set_digital_output(1, OFF)
            set_digital_output(2, ON)
            wait(GRIP_WAIT)

    trigger_estop_and_wait('웍 파지 폭 %d회 연속 비정상 (%.1fmm)'
                           % (WOK_GRIP_WIDTH_RETRY, width))
