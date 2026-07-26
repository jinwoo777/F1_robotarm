"""wok_test4.py `wokking()` 서브루틴에서 추출한 시연(expert) 웍질 궤적.

원본: rokey/wok_test4.py의 반복 구간(SPEED_RATIO=1.0 기준 원속도) —
    movel(P1, vel=[1000,120],  acc=[20000,400])
    movej(J2, ...)                              # 특이점 회피용 관절공간 경유점 — 생략(아래 설명)
    movel(P3, vel=[2000,225],  acc=[75000,900])
    movel(P4, vel=[1000,100],  acc=[20000,400])
    (P1로 복귀, 반복)

각 posx는 [x, y, z, rx, ry, rz] (mm, deg). J2(움직관절 경유점)는 실제 로봇 FK 없이
카테시안으로 정확히 재현할 수 없어 이 근사 궤적에서는 생략했다 — P1→P3 구간의 큰 rz~ry
변화(스윙+손목 플릭)가 이미 "던지기" 동작의 핵심 형상을 담고 있다고 보고, 3점(P1,P3,P4)
왕복만으로 토스 사이클을 근사한다. 실로봇 배치 전 반드시 SPEED_RATIO 단계적 검증 필요.
"""
import numpy as np

# 궤적 파라미터: [x, y, z, rx, ry, rz]
P1 = np.array([698.57, -14.97, 330.02, 178.04, -117.09, 178.50])
P3 = np.array([846.38, -13.77, 336.99, 179.14, -84.31, 178.93])
P4 = np.array([657.22, -18.37, 312.10, 177.96, -104.43, 178.63])

CYCLE = [P1, P3, P4, P1]  # 4 포인트 = 3구간(런칭, 캐치접근, 복귀)

# 각 구간에 "도착"하는 movel 호출에 실제 사용된 vel/acc: [trans(mm/s), rot(deg/s)]
SEGMENT_VEL_LIMIT = [
    np.array([2000.0, 225.0]),  # P1 -> P3 (런칭, 원본 중 가장 빠른 구간)
    np.array([1000.0, 100.0]),  # P3 -> P4 (캐치 접근)
    np.array([1000.0, 120.0]),  # P4 -> P1 (복귀)
]
SEGMENT_ACC_LIMIT = [
    np.array([75000.0, 900.0]),
    np.array([20000.0, 400.0]),
    np.array([20000.0, 400.0]),
]

SEGMENT_TIME_S = 1.0   # 구간당 가정 시간(원본 로그의 샘플링 손실로 정확한 주기 추정 불가 — 가정치)
DT = 0.1               # wok_data_logger.py 의 SAMPLE_HZ=10 과 동일한 스텝
STEPS_PER_SEGMENT = int(round(SEGMENT_TIME_S / DT))
N_SEGMENTS = len(CYCLE) - 1


def segment_limits(seg_idx: int):
    """구간 인덱스(0..N_SEGMENTS-1)에 대한 (vel_limit_6d, acc_limit_6d) 반환.

    [trans,trans,trans, rot,rot,rot] 형태로 축별 동일값 브로드캐스트(단순화된 안전 한계).
    """
    v = SEGMENT_VEL_LIMIT[seg_idx]
    a = SEGMENT_ACC_LIMIT[seg_idx]
    vel_limit = np.array([v[0], v[0], v[0], v[1], v[1], v[1]])
    acc_limit = np.array([a[0], a[0], a[0], a[1], a[1], a[1]])
    return vel_limit, acc_limit
