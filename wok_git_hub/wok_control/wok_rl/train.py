"""웍질 토스 궤적 PPO 학습 진입점.

실행:
    cd rokey/rokey && python3 -m wok_rl.train --iters 300
    (또는 스모크 테스트) python3 -m wok_rl.train --iters 5 --episodes-per-iter 4

산출물(--out-dir, 기본 wok_rl/runs/<timestamp>/):
    reward_curve.png   — 학습 곡선
    trajectory.png      — 학습된 궤적 vs 원본 웨이포인트(P1,P3,P4) 비교
    learned_trajectory.csv — 학습된 정책의 결정론적 1사이클 궤적(x,y,z,rx,ry,rz)
    policy.pt           — 정책 가중치 체크포인트
"""
import argparse
import csv
import datetime
import os

import numpy as np
import torch

from . import waypoints as W
from .env import WokTossEnv
from .ppo import PPO


def rollout_deterministic(env: WokTossEnv, ppo: PPO):
    """탐색 노이즈 없이(평균 행동) 1사이클 굴려서 궤적/보상 기록."""
    obs = env.reset()
    positions, rewards, infos = [env.pos.copy()], [], []
    for _ in range(env.horizon):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean_action = torch.tanh(ppo.ac.actor_mean(obs_t)).numpy()
        obs, reward, terminated, truncated, info = env.step(mean_action)
        positions.append(env.pos.copy())
        rewards.append(reward)
        infos.append(info)
        if terminated or truncated:
            break
    return np.array(positions), np.array(rewards), infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--episodes-per-iter", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = WokTossEnv(seed=args.seed)
    ppo = PPO(env.obs_dim, env.act_dim)

    steps_per_iter = env.horizon * args.episodes_per_iter
    reward_history = []

    for it in range(1, args.iters + 1):
        batch = ppo.collect_rollout(env, steps_per_iter)
        ppo.update(batch)
        mean_ret = float(np.mean(batch["ep_returns"])) if batch["ep_returns"] else float("nan")
        reward_history.append(mean_ret)
        if it % args.log_every == 0 or it == 1:
            print(f"iter {it:4d}/{args.iters}  mean_episode_return={mean_ret:8.3f}")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    positions, rewards, infos = rollout_deterministic(env, ppo)
    final_trans_err = infos[-1]["trans_err"] if infos else float("nan")
    final_rot_err = infos[-1]["rot_err"] if infos else float("nan")
    print(f"\n학습 완료 -> {out_dir}")
    print(f"최종 사이클 도달 오차: trans={final_trans_err:.2f}mm, rot={final_rot_err:.2f}deg")
    print(f"결정론적 롤아웃 총 보상: {rewards.sum():.3f}")

    torch.save(ppo.ac.state_dict(), os.path.join(out_dir, "policy.pt"))

    with open(os.path.join(out_dir, "learned_trajectory.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "t_s", "x", "y", "z", "rx", "ry", "rz"])
        for i, p in enumerate(positions):
            writer.writerow([i, round(i * env.dt, 3)] + [round(v, 3) for v in p])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(reward_history)
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean episode return")
        ax.set_title("WokTossEnv PPO training curve")
        fig.savefig(os.path.join(out_dir, "reward_curve.png"), dpi=120)
        plt.close(fig)

        demo_xyz = np.array([p[:3] for p in W.CYCLE])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(demo_xyz[:, 0], demo_xyz[:, 2], "o--", label="demo waypoints (P1->P3->P4->P1)")
        axes[0].plot(positions[:, 0], positions[:, 2], "-", label="learned trajectory")
        axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("z (mm)"); axes[0].legend(); axes[0].set_title("XZ path")

        t_axis = np.arange(len(positions)) * env.dt
        axes[1].plot(t_axis, positions[:, 4], label="learned ry(t)")
        axes[1].axhline(W.P1[4], color="gray", linestyle=":", label="demo ry at P1/P3/P4")
        axes[1].axhline(W.P3[4], color="gray", linestyle=":")
        axes[1].axhline(W.P4[4], color="gray", linestyle=":")
        axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("ry (deg)"); axes[1].legend(); axes[1].set_title("wrist flip over cycle")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "trajectory.png"), dpi=120)
        plt.close(fig)
    except ImportError:
        print("matplotlib 미설치 — 플롯 생략, CSV/체크포인트만 저장됨")


if __name__ == "__main__":
    main()
