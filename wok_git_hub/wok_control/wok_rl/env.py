"""WokTossEnv — 웍질 1사이클(P1->P3->P4->P1)에 대한 경량 카테시안 운동학 시뮬레이션 환경.

실제 로봇 동역학/접촉/충돌은 모델링하지 않는다(물리엔진 아님). 위치를 속도로 적분하는
점질량(point-mass) 근사이며, 목적은 "구간 끝점(원본 웨이포인트)은 고정한 채 구간 사이의
속도/가속 프로파일(경로 형상)을 보상 기반으로 탐색"하는 것이다. 실로봇 배치 전 반드시
사람 검증 + SPEED_RATIO 단계적 테스트가 필요하다(이 코드는 그 전 단계의 오프라인 탐색용).

Gym 스타일 API(reset/step)를 직접 구현 — gymnasium 미설치 환경에서도 실행되도록 의존성 없음.
"""
import numpy as np

from . import waypoints as W


class WokTossEnv:
    # 보상 가중치 — 사용자 우선순위: 토스 품질(런칭/캐치 역학) > 안전 > 사이클타임/시연유사도
    W_PROGRESS = 0.01        # 다음 목표점까지 거리 감소 shaping
    W_ARRIVAL = 2.0          # 구간 종료 시 목표점 도달 보너스(=원본 웨이포인트 고정, 안전/호환성)
    W_LAUNCH = 0.05          # 런칭 구간(P1->P3) 회전(ry) 각속도 + 병진 속도 보상 = 토스 품질 핵심
    W_CATCH_SMOOTH = 1.0     # 사이클 종료 시점 잔류 속도 페널티 = 부드러운 캐치
    W_SAFETY = 5.0           # 속도/가속 한계 초과(클리핑량) 페널티
    W_TIME = 0.01            # 스텝당 소폭 페널티(사이클타임 최소화, 낮은 우선순위)

    ARRIVAL_TOL_TRANS = 5.0   # mm
    ARRIVAL_TOL_ROT = 2.0     # deg

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.dt = W.DT
        self.steps_per_segment = W.STEPS_PER_SEGMENT
        self.n_segments = W.N_SEGMENTS
        self.horizon = self.steps_per_segment * self.n_segments
        self.obs_dim = 6 + 6 + 6 + 1 + self.n_segments  # pos, vel, err, phase_frac, segment_onehot
        self.act_dim = 6
        self.reset()

    def reset(self):
        self.pos = W.CYCLE[0].copy()
        self.vel = np.zeros(6)
        self.seg_idx = 0
        self.step_in_seg = 0
        self.t = 0
        return self._obs()

    def _current_target(self):
        return W.CYCLE[self.seg_idx + 1]

    def _obs(self):
        target = self._current_target()
        err = target - self.pos
        phase_frac = np.array([self.step_in_seg / self.steps_per_segment])
        seg_onehot = np.eye(self.n_segments)[self.seg_idx]
        return np.concatenate([self.pos, self.vel, err, phase_frac, seg_onehot]).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        vel_limit, acc_limit = W.segment_limits(self.seg_idx)

        vel_cmd = action * vel_limit
        max_dvel = acc_limit * self.dt
        vel_cmd = np.clip(vel_cmd, self.vel - max_dvel, self.vel + max_dvel)
        clip_amount = np.abs(vel_cmd - action * vel_limit).sum()  # 가속 한계로 잘려나간 양(안전 페널티용)

        prev_pos = self.pos.copy()
        self.pos = self.pos + vel_cmd * self.dt
        self.vel = vel_cmd

        target = self._current_target()
        trans_err = np.linalg.norm((target - self.pos)[:3])
        rot_err = np.linalg.norm((target - self.pos)[3:])
        prev_trans_err = np.linalg.norm((target - prev_pos)[:3])
        prev_rot_err = np.linalg.norm((target - prev_pos)[3:])

        reward = 0.0
        # 1) 목표점 접근 shaping (거리 감소량에 비례)
        reward += self.W_PROGRESS * ((prev_trans_err - trans_err) + 10.0 * (prev_rot_err - rot_err))

        # 2) 런칭 구간(0번, P1->P3): 손목 회전(ry) 각속도 + 병진 속도 = "토스 힘" 보상
        if self.seg_idx == 0:
            reward += self.W_LAUNCH * (abs(self.vel[4]) + np.linalg.norm(self.vel[:3]) / 100.0)

        # 3) 안전: 가속 한계로 클리핑된 양에 비례한 페널티
        reward -= self.W_SAFETY * (clip_amount / (np.abs(vel_limit).sum() + 1e-6))

        # 4) 시간 페널티
        reward -= self.W_TIME

        self.step_in_seg += 1
        self.t += 1
        terminated = False
        if self.step_in_seg >= self.steps_per_segment:
            # 구간 종료: 원본 웨이포인트 도달 보너스(멀수록 큰 페널티) — 끝점을 실제 wok_test4.py
            # 웨이포인트에 고정시켜 이후 실로봇 이식 가능성을 유지하는 핵심 항
            reward -= self.W_ARRIVAL * (trans_err / 50.0 + rot_err / 10.0)

            is_last_segment = self.seg_idx == self.n_segments - 1
            if is_last_segment:
                # 사이클 종료: 잔류 속도가 작을수록(=부드러운 캐치) 보상
                reward -= self.W_CATCH_SMOOTH * (np.linalg.norm(self.vel[:3]) / 200.0
                                                  + np.linalg.norm(self.vel[3:]) / 20.0)
                terminated = True
            else:
                self.seg_idx += 1
                self.step_in_seg = 0

        truncated = self.t >= self.horizon
        info = {"trans_err": trans_err, "rot_err": rot_err, "segment": self.seg_idx}
        return self._obs(), reward, terminated, truncated, info
