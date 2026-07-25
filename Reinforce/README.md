# Wok Sim

재료의 **합산 질량**을 입력으로 받아 open-loop 웍질 궤적 파라미터를 한 번
선택하고, MuJoCo에서 같은 형태의 웍질을 5회 연속 실행해 혼합·유실·비행 및
궤적 품질을 평가하는 Python 3.11+ 프로젝트다.

> 주의: 기본 YAML의 수치는 소프트웨어 동작 확인용 demo 값이다. 실제 M0609의
> 안전 한계, 허용 속도 또는 가속도를 뜻하지 않는다. 생성된 궤적을 검증 없이
> 실제 로봇에서, 특히 고속으로 실행하면 안 된다. 이 저장소는 실제 로봇에
> 연결하거나 명령을 보내지 않는다.

## 모델과 open-loop의 의미

기본 legacy profile에서 정책 observation의 기본값은 `[total_mass_kg]`이고,
선택적으로 평균 반지름과 표준편차를 더할 수 있다. 정책은 에피소드 시작 때
아래 7개 정규화 action을 딱 한 번 출력한다.

1. insertion distance
2. lift height
3. backward distance
4. pitch amplitude
5. cycle time
6. insert phase ratio
7. catch phase ratio

실행 중 입자 위치·속도, 카메라, 가속도계 또는 토크 센서를 정책에 되먹임하지
않는다. 여기서 open-loop는 **외부 궤적 수정이 없다**는 뜻이며, MuJoCo의 접촉
해석과 실제 로봇 내부 servo까지 없앤다는 뜻은 아니다. Gym 환경은 한 번의
`step(action)`이 전체 5회 동작을 수행하고 곧바로 종료되는 one-step contextual
RL(contextual bandit에 가까운 구조)이다.

팬은 궤적을 직접 따르는 mocap/kinematic rigid body다. 팬 질량 0.700 kg은
metadata에 보존하지만, 알려지지 않은 관성모멘트를 만들어내지 않으며 현
입자 시뮬레이션에는 쓰지 않는다. 재료 충돌도 팬 궤적을 바꾸지 않는다. 기본
legacy profile의 입자는 동일 밀도의 구로 단순화한다. N과 밀도를 고정한 채
반지름 전체 scale을 조정해 목표 합산 질량을 정확히 맞춘다.

따라서 legacy profile에서 합산 질량을 바꾸면 입자 크기도 함께 바뀐다.
관측되는 차이가 순수한 질량 효과인지 크기 효과인지 완전히 분리할 수 없다는
한계가 있다. 아래 볶음밥 profile은 크기와 종류를 명시적으로 고정한 별도
입자 모델을 사용한다.

## 볶음밥 pan-only profile

`configs/fried_rice.yaml`은 M0609 모델, URDF, SDK 또는 ROS 없이 팬과 재료만
시뮬레이션하는 profile이다. 100% 양은 다음 재료를 각각 20개, 합계 60개다.
환경은 공통 종별 개수 `n` 20~40, 즉 총 60~120개를 지원하지만 현재 학습은
`n=[20,30,40]`인 총 60/90/120개 조건만 각각 50 episode씩 사용한다. 세 종류의
비율은 항상 1:1:1이다. 타원체 크기는 현재
**반축(semi-axis)** 으로 해석한다. 사용자가
말한 0.5/0.2/0.2 cm가 전체 축 길이라는 뜻이었다면 설정값을 절반으로 바꿔야
한다.

입자는 시작할 때 팬 중심 반경 5 cm 안에 충돌 없이 모아 생성한다. 60/90/120개
조건에서는 필요한 경우 높이 방향으로 층을 추가하며, 생성 seed가 같으면 초기
배치도 재현된다.

| 종류 | MuJoCo 형상과 크기 | 개당 질량 | 20 cm 낙하 목표 반발 높이 | 이상적 반발계수 |
|---|---|---:|---:|---:|
| 큰 구 | sphere, 반지름 0.5 cm | 1.8~2.2 g 독립 균등분포 | 3 cm | 0.387 |
| 작은 구 | sphere, 반지름 0.3 cm | 0.8 g | 1 cm | 0.224 |
| 타원체 | ellipsoid, 반축 0.5/0.2/0.2 cm | 0.2 g | 1.5 cm | 0.274 |

큰 구의 질량만 입자별로 독립적으로 달라진다. 100% 양의 nominal 합산 질량은
60 g이고 전체 domain-randomization support는 56~128 g이다. 정책 observation은
`[실제 총질량 정규화, 종별 개수 정규화]` 2D이며 둘 다 설정 범위에서
`[-1,1]`로 변환된다. 실제 kg, 총 개수, 종별 개수와 100% 대비 양은 `info`와
CSV에 원 단위로 남는다. `reset(options={"count_per_type": 30})`처럼 종별
개수를 고정해 총 90개 조건을 재현할 수도 있다.

낙하 직전과 반발 직후의 위치에너지 비로부터
`e_ideal = sqrt(h_rebound / h_drop)`를 계산하고,
`zeta = -ln(e) / sqrt(pi^2 + ln(e)^2)`로 이론 damping ratio도 기록한다.
MuJoCo의 discrete soft contact에는 이론값을 그대로 쓰지 않고 MuJoCo 3.10,
2 ms timestep, 6 ms contact time constant의 수평 plane 낙하 simulation에서
목표 첫 반발 높이에 맞도록 damping ratio를 binary search한 값을 사용한다.
보정은 다음 명령으로 재현할 수 있다.

```bash
python scripts/calibrate_particle_bounce.py
```

이 보정도 이상적인 단일 수직 충돌의 근사다. 실제 pan 안의 반발 높이는
compound proxy, 마찰, 입자 자세·회전 및 solver 설정의 영향을 함께 받으므로
목표 높이가 모든 상황에서 보장되지는 않는다. 실제 재료 적용 전에는 같은
20 cm drop test를 profile별로 반복해 parameter를 다시 식별해야 한다.

### 4단 teaching과 4D action

한 cycle은 다음 네 단계를 이 순서로 수행하고 기본값은 5 cycle이다.

1. 시작 위치에서 팬을 먼저 +pitch 30도로 기울인다.
2. pitch 30도를 유지한 채 +x 기준 아래쪽 45도 방향으로 25 cm 하강한다.
3. 하강 위치를 유지한 채 학습된 lift 각도만큼 pitch를 복원한다.
4. 시작 위치로 돌아오면서 pitch도 시작 각도로 동시에 복원한다.

볶음밥 정책은 기존 legacy 7D action 대신 다음 4D 정규화 action을 한 번
출력한다.

1. `descent_angle`: +x 기준 아래쪽 하강 경로 각도
2. `pan_tilt_angle`: 하강 중 유지할 +pitch
3. `descent_speed`: 하강의 목표 최대 선속도
4. `lift_angle`: 삽입 위치에서 팬을 들어 올릴 pitch 복원량

하강각 범위는 35~55도, 팬 tilt 범위는 20~40도, 속도 범위는
0.38~0.48 m/s, lift 각도 범위는 0~30도다. 이동 거리는 25 cm, 목표 최대
각속도는 1.20 rad/s로 고정한다. 중앙 action은 lift 15도이며 하강 끝점은 시작점 기준
`x≈+0.1768 m`, `z≈-0.1768 m`이고 pitch는 하강 내내 +30도다. +pitch는
전방인 +x 쪽 팬 가장자리를 낮춰 입자가 앞쪽에 모이도록 한다. 이 방향과
workspace는 pan-only simulation 계약이며 실제 테이블 또는 로봇 간섭을
통과했다는 뜻이 아니다.

각 phase는 `s(u)=10u³-15u⁴+6u⁵` minimum-jerk 곡선을 사용해 waypoint 사이를
단조롭게 이동한다. 따라서 요구 거리와 tilt 각도를 넘는 spline overshoot가
없고 단계 경계에서 속도·가속도가 0이다. 각 phase 시간은 목표 속도 외에도
provisional 가속도·jerk gate를 만족하도록
`max(1.875D/v, sqrt(5.7735D/a), cbrt(60D/j))`로 정한다. 첫 phase는 선행 tilt,
둘째는 45도 하강, 셋째는 학습된 lift, 넷째는 위치와 남은 pitch의 동시 원복이다.
중앙 action은 하강각 45도, 팬 tilt 30도, 속도 0.43 m/s, lift 15도이며 한 cycle 약
4.265초, 다섯 cycle 약 21.327초다. 16개 action corner의 다섯-cycle 길이는
약 17.73~24.55초다. action은 episode 시작 때 결정된 뒤 전체 실행 동안
고정되며 입자나 영상 feedback으로 수정되지 않는다.

action shape이 기존 3D에서 4D로 바뀌었으므로 기존 checkpoint는 새 환경과
호환되지 않는다. 최초 확인은 checkpoint 없이 중앙 action으로 실행하고 학습은
새 `fried_rice_sac_150_clustered_lift4d_bonus010` 경로에서 처음부터 시작한다.

SAC의 `MlpPolicy`를 사용한다. 입력은 정규화 질량·개수의 저차원 수치이고
출력도 연속 4D action이므로 작은 MLP가 현재 문제에 가장 직접적이다. CNN은 영상
observation이 없고 Transformer는 긴 관측 sequence가 없는 현재 one-step
구조에서 이점보다 계산량이 크다. SAC는 연속 action을 직접 다루고 replay
buffer의 표본을 재사용할 수 있다는 점 때문에 선택했다.

### episode random walk

`BoundedEpisodeRandomWalk`는 정규화 action 공간의 설정된 경계에서 Gaussian
step을 적용하고, 경계를 넘은 값은 clipping 대신 반사한다. 현재 action은
`advance_episode()`를 호출하기 전까지 변하지 않으므로 한 cycle 중 진동이나
실행 중 feedback을 만들지 않는다. 같은 seed면 proposal 열도 재현된다.

```python
from wok_sim.exploration import BoundedEpisodeRandomWalk

walk = BoundedEpisodeRandomWalk(4, step_std=[0.05, 0.05, 0.03, 0.05], seed=7)
action_for_episode_0 = walk.current_proposal().action
action_for_episode_1 = walk.advance_episode().action
```

볶음밥 설정은 `training.random_walk.enabled: true`로 하강각, 팬 각도,
0.38~0.48 m/s 하강 속도와 0~30도 lift 각도를 모두 SAC의 additive
random-walk noise에 연결한다.
`step_std`는 episode 사이의 정규화 변화량, `bound`는 noise 자체의 양·음
경계이며, SB3가 episode 종료 때 보내는
`reset()`에도 walk 상태를 유지한다. 이 환경에서는 한 `env.step()`이 곧 전체
5-cycle episode이므로 noise는 다음 episode의 action을 고를 때만 한 번
갱신되고 MuJoCo physics step 중에는 호출되지 않는다. deterministic 평가는
이 noise를 사용하지 않는다.

### 실행과 학습

볶음밥 profile은 `--mass-g` 대신 `--count-per-type`으로 양을 고정한다.

```bash
python -m wok_sim.cli check-assets --config configs/fried_rice.yaml
python -m wok_sim.cli rollout \
  --config configs/fried_rice.yaml --count-per-type 30 --seed 1 --headless
python -m wok_sim.cli baseline \
  --config configs/fried_rice.yaml --episodes 20 --strategy random_walk
python -m wok_sim.cli train \
  --config configs/fried_rice.yaml \
  --checkpoint checkpoints/fried_rice_sac_150_clustered_lift4d_bonus010/policy --timesteps 150
python -m wok_sim.cli evaluate \
  --config configs/fried_rice.yaml \
  --checkpoint checkpoints/fried_rice_sac_150_clustered_lift4d_bonus010/policy.zip --episodes 30 \
  --count-per-type 20 --count-per-type 30 --count-per-type 40
python -m wok_sim.cli export-trajectory \
  --config configs/fried_rice.yaml \
  --checkpoint checkpoints/fried_rice_sac_150_clustered_lift4d_bonus010/policy.zip \
  --output results/fried_rice_trajectory.csv
python -m wok_sim.cli check-motion-contract \
  --config configs/fried_rice.yaml
python -m wok_sim.cli export-doosan-plan \
  --config configs/fried_rice.yaml \
  --function movel --output results/fried_rice_movel_plan.json
```

현재 설정은 one-step 환경의 `150 timestep = 150 episode`를 학습한다. 종별
개수 schedule `[20,30,40]`은 총 입자 60/90/120개에 해당하며 각 조건을 정확히
50회씩 배정한다. 논리 코어 12개 중 절반인 MuJoCo 환경 6개를 Windows `spawn`
방식으로 병렬 실행하고,
30회 warm-up 뒤 업데이트한다. one-step contextual 문제에 맞춰 `gamma=0`,
physics sample 하나당 gradient update 4회를 사용한다. lift 보상은 최종 유출되지
않은 유효 입자 한 알당 0.10점이다. 자동 평가는 끄고 30
episode마다 중간 모델과 replay buffer를 저장한다. 150회 완료 시 최종
`policy.zip`을 저장하며, 이는 수렴 보장이 아니라 새 teaching motion의
physics/reward/학습 배선과 초기 reward 추세를 확인하는 파일럿이다.

짧은 smoke rollout과 기능 검증에는 Modal 또는 Colab이 필요하지 않다.
MuJoCo의 60~120입자 접촉 계산은 CPU 작업이고 policy도 작은 MLP이므로 T4를
추가해도 episode 물리 계산은 빨라지지 않는다. 현재 설정도 이 이유로
`training.device: cpu`를 사용한다. action에 따라 다섯-cycle trajectory는 약
17.73~24.55초이고 입자 수도 달라져 episode 벽시계 시간의 편차가 크다. 이전
3D teaching의 실행 시간과 mixing 수치는 새 4D motion의 예상치로 재사용하지
않고, 최초 1회 확인과 150회 학습에서 새로 기록한다. 이 저장소 자체는 외부
학습 서버나 API 연결을 요구하지 않는다.

### M0609 경계와 오프라인 검증

`robot.enabled: false`인 볶음밥 pan-only 실행 경로는 M0609 validator,
Pinocchio, Doosan SDK 또는 ROS를 import하지 않고 실제 로봇에 연결하거나
명령을 보내지 않는다. 별도 `m0609_motion_contract` 모듈은 생성된 Cartesian
trajectory를 속도·가속도·jerk cap과 비교하고 `movesx` 또는 `movel` 형태의
**비실행 parameter template**만 JSON/CSV/Python data로 만들 수 있다.
Python 출력에도 API import나 함수 호출은 들어가지 않는다. 어느 cap이든
넘으면 export를 차단하고 같은 경로 형상을 유지하는 데 필요한 uniform
time-scale을 report할 뿐, 한계를 높이거나 trajectory를 몰래 retime하지
않는다.

볶음밥 profile의 offline cap은 선속도 0.50 m/s, 각속도 1.30 rad/s,
선가속도 1.50 m/s², 각가속도 2.70 rad/s², 선 jerk 16.0 m/s³,
각 jerk 22.0 rad/s³다. 16개 4D action corner에서 측정된 최댓값은 각각
약 0.4617 m/s, 1.0231 rad/s, 1.4000 m/s², 2.4623 rad/s²,
14.329 m/s³, 20.0 rad/s³다. 선속도 cap은 공개된 M0609의 약 1 m/s
TCP 최대 속도보다 낮춘 reference이고, 나머지도 제조사 한계가 아닌 초기
engineering 값이다. cap을 통과해도 safety status는 항상
`unverified_without_urdf_teaching`이다.
공개 reference와 함수 규약은 Doosan의
[M0609 사양](https://manual.doosanrobotics.com/en/user-manual/3.3.0/1-m-h-series/m0609),
[`movesx`](https://manual.doosanrobotics.com/en/programming-manual/3.5.0/publish/movesx),
[`movel`](https://manual.doosanrobotics.com/en/programming-manual/3.6.0/publish/movel)
문서를 기준으로 했으며, 문서 수치를 현재 장비의 safety approval로 해석하면
안 된다.

실제 M0609 적용 전에 최소한 실제 `T_base_tcp_teach`, `T_tcp_pan`, 장착물
질량·무게중심·관성, URDF/controller joint 제한을 넣고 sequential IK,
singularity, self/environment collision, joint speed/acceleration 및
controller interpolation을 다시 검증해야 한다. 팬 0.7 kg과 최대 약 128 g 재료만의
합보다 실제 손잡이·adapter 하중이 크며, 공개 payload 수치만으로 wrist
moment나 동적 실행 가능성을 보장할 수 없다.

## 설치

Python 3.11 이상에서:

```bash
cd Reinforce
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

저장소의 `uv.lock`을 사용하는 경우 `uv sync --extra dev`로 같은 dependency
해결 결과를 재현할 수 있다. 개발 검증 환경은 Python 3.11.15, MuJoCo 3.10.0,
Gymnasium 1.3.0, Stable-Baselines3 2.9.0, CPU PyTorch 2.13.0 조합이며
dependency checker로 설치 충돌이 없음을 확인했다.

사용자 URDF로 robot validation까지 실행할 때만 Pinocchio extra를 설치한다.

```bash
python -m pip install -e ".[robot,dev]"
```

모든 내부 단위는 SI(m, kg, s, rad)다. CLI의 `--mass-g`만 gram을 즉시 kg로
변환하는 편의 입력이다.

## STL과 collision proxy

`Wok.stl` 또는 사용자의 STL 경로를 `pan.stl_path`에 지정하고 단위를 확인해
`pan.stl_scale`을 직접 입력한다. 예를 들어 STL 수치가 mm라면 `0.001`일 수
있지만 프로그램은 단위를 자동 확정하지 않는다.

```bash
python -m wok_sim.cli check-assets --config configs/default.yaml
```

명령은 vertex/face 수, scale 적용 후 bounding box와 watertight 여부를
출력한다. STL은 visual mesh에만 쓴다. 오목한 팬을 단일 convex hull collision
geometry로 바꾸지 않고, 바닥과 원주 방향 wall/rim primitive로 구성한 조정
가능한 compound proxy를 사용한다.

`configs/test.yaml`은 `use_procedural_demo: true`를 명시한 빠른 테스트 전용
팬이다. 실제 STL 경로가 틀렸을 때 procedural 팬으로 조용히 대체하지 않는다.
procedural 성공은 사용자 STL의 접촉 형상이 검증됐다는 의미가 아니다.

## 실행

실제 STL을 쓰는 headless 고정 질량 rollout:

```bash
python -m wok_sim.cli rollout \
  --config configs/default.yaml --mass-g 150 --seed 1 --headless
```

명시적 테스트 팬 rollout과 random baseline:

```bash
python -m wok_sim.cli rollout \
  --config configs/test.yaml --mass-g 20 --seed 1 --headless
python -m wok_sim.cli baseline --config configs/test.yaml --episodes 10
```

SAC 학습과 평가:

```bash
python -m wok_sim.cli train --config configs/default.yaml
python -m wok_sim.cli evaluate \
  --config configs/default.yaml --checkpoint checkpoints/sac_wok.zip
```

학습 중에는 `training.evaluation_interval`마다 별도 deterministic 평가를
실행하고 `training.evaluation_episodes`개 episode의 결과와 best model을
해당 실행 결과 디렉터리의 `evaluation/`에 저장한다. `0`이면 주기 평가를
비활성화한다.

학습된 정책(또는 `--checkpoint` 생략 시 중앙 고정 action)의 궤적 export:

```bash
python -m wok_sim.cli export-trajectory \
  --config configs/default.yaml --mass-g 150 \
  --checkpoint checkpoints/sac_wok.zip \
  --output results/trajectory_150g.csv
```

CSV/JSON/NPZ에는 timestamp, pan/TCP pose, 선속도·각속도,
선가속도·각가속도와 jerk가 포함된다. robot validation이 실행되면 joint
position/velocity/acceleration도 포함할 수 있다. `T_tcp_pan`이 없을 때의 TCP
열은 시뮬레이션 편의를 위한 identity frame 가정으로 metadata에 표시되며 실제
로봇 TCP로 간주하면 안 된다.

## 궤적과 metric

`wok_frame`은 +x가 앞으로 꽂는 방향, +y가 좌측, +z가 위쪽이다. 움직임은
x-z translation과 y축 pitch만 허용한다. 볶음밥 profile은 먼저 +pitch
20~40도로 기울인 뒤 그 각도를 유지하면서 +x 기준 아래쪽 35~55도로 하강한다.
중앙 teaching은 pitch 30도와 하강각 45도이며, 다음 phase에서 같은 하강 위치를
유지하고 학습된 0~30도 lift만큼 pitch를 복원한 뒤 위치와 각도를 함께 원복한다.

볶음밥 profile은 P0~P4의 네 phase를 minimum-jerk spline으로 연결해 5회
반복한다. 방향이 바뀌는 각 phase 경계에서 속도와 가속도는 0이며 별도 dwell은
없다. analytic 1~3차 미분으로 속도, 가속도, jerk를 계산하고
workspace/평면/미분 limit 위반 action은 입자 시뮬레이션 전에 invalid 처리한다.
legacy profile의 P0~P5 global spline 계약은 별도로 유지된다.

waypoint와 workspace 검사는 `wok_frame`에서 수행하고, MuJoCo 실행·TCP 변환과
export에는 base/world frame으로 변환한 pose를 사용한다. 고정된 roll/yaw
상태에서 pitch가 변하더라도 실제 각속도 벡터는 한 RPY 성분의 단순 미분이
아닐 수 있으므로, export의 각속도·각가속도·각 jerk는 SO(3) 강체 운동학으로
계산한다. TCP 원점이 pan 원점과 떨어져 있으면 선속도·가속도·jerk에도 회전
오프셋 항을 포함한다.

혼합도는 world x-z가 아닌 pan-local 바닥 좌표를 grid로 나눈 뒤 초기 quadrant
label의 정규화 조건부 엔트로피 `H(C|B)/H(C)`로 측정한다. reward는 초기 대비
개선량 `final - initial`만 쓴다. 볶음밥 profile은 각 cycle의 하강 완료 뒤
학습된 lift와 원위치 복귀 구간에서, pan과 비접촉이고 grain 윗부분이 rim보다
1 mm를 초과하면서 최종 유출되지 않은 grain에 episode당 한 번 0.10점의 lift
보너스를 준다. 같은 grain의 반복 통과는 중복 계산하지 않고, 밖으로 던진
grain으로 점수를 얻을 수 없게 한다. 유실은 최종 pan-local 상태와 영구 spill
boundary 통과를 함께 보며, 잠깐 비행했다가 팬으로 돌아온 입자를 유실로 세지
않는다. 볶음밥
유출 감점은 질량/개수 비율 중 큰 값 `r`에 대해 `-(12r + 40r²)`라서 유출이
심해질수록 한 개 추가 유출의 감점도 커진다. 접촉 해제와 상대 상향 속도로
takeoff를 찾아 입자별 world/relative 최대 높이, 지속시간과 상대 속도 기반
launch angle도 별도로 기록한다.

## TEACHING pose와 M0609

`configs/user.yaml.example`을 `configs/user.yaml`로 복사해 아래 현장 값을
넣는다. example의 `base_config: default.yaml`이 나머지 demo 설정을 상속하므로
현장 override만 유지할 수 있다.

- `q_teach`: URDF model 차원과 맞는 초기 관절각(rad)
- `T_base_tcp_teach`: base 기준 teaching TCP 4x4 transform
- `T_tcp_pan`: TCP에서 pan frame으로의 고정 4x4 transform
- `T_base_wok`: 별도 wok frame 4x4 transform
- `urdf_path`, `base_link`, `tcp_link`
- URDF에 없는 joint velocity/acceleration 제한

초기 pan pose의 우선순위는
`T_base_tcp_teach @ T_tcp_pan`, `pan.initial_pose`, wok-local
`trajectory.start_*`, 시뮬레이션 기본값 순이다. teaching transform을 넣을
때는 example처럼 중복된 `pan.initial_pose`와 `trajectory.start_*`를 `null`로
두거나, 같은 pose를 각각 base 및 wok frame에 맞게 정확히 입력해야 한다.
중복 값이 tolerance 밖에서 충돌하면 시작 순간의 pan teleport를 막기 위해
설정 오류로 중단한다. 모든 4x4 transform은 마지막 행, 회전 직교성 및
determinant까지 검사한다.

URDF 또는 teaching 값이 없고 `required: false`면 상태는 명시적으로
`not_evaluated`다. `required: true`면 누락을 오류로 처리한다. 제한값을
임의로 M0609 값이라고 하드코딩하지 않는다. URDF가 제공되면 teaching q를 첫
seed로 쓰고 이전 q를 다음 sequential IK seed로 사용해 branch continuity,
URDF position/velocity limit, 사용자가 준 acceleration limit, singularity와
선택적 self-collision을 검사한다.

검증 범위는 기구학적 실행 가능성까지다. 팬 관성모멘트와 실제 장비 모델이
없으므로 다음은 검증하지 못한다.

- 실제 관절 torque와 motor 부하
- 실제 servo tracking error
- 손잡이 진동 및 payload 관성에 따른 overshoot
- 현장 environment collision과 실제 재료의 비구형 거동

또한 현재 팬은 MuJoCo의 mocap/kinematic body로 pose를 직접 지정한다. 이
방식은 요구된 “입자가 팬 궤적을 바꾸지 않는” open-loop 모델에는 적합하지만,
팬 자체의 generalized velocity·관성 및 접촉 반작용을 동적으로 적분하지
않는다. 따라서 고속 접촉에서 전달되는 에너지, 실제 팬/재료 충격력과 모터
부하는 정량적으로 검증된 값이 아니다. compound proxy와 solver parameter의
민감도 검증이 실제 장비 적용 전에 필요하다.

## 결과

각 CLI 실행은 `results/YYYYMMDD_HHMMSS_xxxxxx/` 아래 effective config와
metadata를 저장한다. episode를 수행하는 명령은 summary와 필요시 particle
trajectory도 저장한다. episode summary는 seed/질량/입자 조건, 물리 action,
최대 Cartesian 미분값과 cost, robot status, 초기·최종 혼합도, 유실
count/mass, 비행 높이·각도 통계 및 reward를 포함한다.
`logging.save_csv`, `save_npz`, `save_particle_trajectories`로 큰 출력의 저장을
제어한다. `save_video`는 현재 `rgb_array` 실행 중 RGB frame 수집까지만
지원하며, CLI에서 MP4로 인코딩하는 기능은 아직 제공하지 않는다.

궤적 그림은 팬 중심을 빨간색, collision proxy의 실제 rim centerline에 있는
뒤쪽(-x)·앞쪽(+x) 끝점을 파란색으로 표시한다. 세 궤적 모두 같은 시간축에서
0.1초 간격으로 보간한 점을 사용하므로, 제자리 lift 때 중심이 겹쳐도 양끝의
회전 궤적으로 팬의 들림을 확인할 수 있다.

## 개발 검증

```bash
pytest -q
python -m wok_sim.cli smoke-test --config configs/test.yaml
```

자동 테스트는 exact mass와 seed 재현성, 선행 30도 tilt·45도 하강·0~30도 lift,
4D action corner, 60/90/120개 schedule의 각 50회 배정, spline 연속성과 5회
반복, mixing/spill/flight, Gym API, robot
비활성 동작과 export schema를 검사한다. 실제 로봇 실행 기능은 의도적으로
구현하지 않는다.
