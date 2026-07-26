"""최소 구현 PPO (clip objective + GAE) — 하나의 환경에서 순차 롤아웃 수집."""
import numpy as np
import torch
import torch.nn as nn

from .policy import ActorCritic


class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_ratio=0.2, epochs=10, minibatch_size=64, entropy_coef=0.001,
                 device="cpu"):
        self.device = device
        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef

    def collect_rollout(self, env, n_steps: int):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        ep_returns, ep_infos = [], []

        obs = env.reset()
        ep_ret = 0.0
        for _ in range(n_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, logp, value = self.ac.act(obs_t)
            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            obs_buf.append(obs)
            act_buf.append(action_np)
            logp_buf.append(logp.item())
            rew_buf.append(reward)
            val_buf.append(value.item())
            done_buf.append(terminated or truncated)
            ep_ret += reward

            obs = next_obs
            if terminated or truncated:
                ep_returns.append(ep_ret)
                ep_infos.append(info)
                ep_ret = 0.0
                obs = env.reset()

        with torch.no_grad():
            last_val = self.ac.value(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).item()

        return dict(
            obs=np.array(obs_buf, dtype=np.float32),
            act=np.array(act_buf, dtype=np.float32),
            logp=np.array(logp_buf, dtype=np.float32),
            rew=np.array(rew_buf, dtype=np.float32),
            val=np.array(val_buf, dtype=np.float32),
            done=np.array(done_buf, dtype=np.bool_),
            last_val=last_val,
            ep_returns=ep_returns,
        )

    def _gae(self, rew, val, done, last_val):
        n = len(rew)
        adv = np.zeros(n, dtype=np.float32)
        lastgae = 0.0
        for t in reversed(range(n)):
            next_val = last_val if t == n - 1 else val[t + 1]
            next_nonterminal = 0.0 if done[t] else 1.0
            delta = rew[t] + self.gamma * next_val * next_nonterminal - val[t]
            lastgae = delta + self.gamma * self.lam * next_nonterminal * lastgae
            adv[t] = lastgae
        ret = adv + val
        return adv, ret

    def update(self, batch):
        adv, ret = self._gae(batch["rew"], batch["val"], batch["done"], batch["last_val"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(batch["obs"], device=self.device)
        act = torch.as_tensor(batch["act"], device=self.device)
        logp_old = torch.as_tensor(batch["logp"], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)

        n = obs.shape[0]
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                logp, entropy, value = self.ac.evaluate(obs[mb], act[mb])
                ratio = torch.exp(logp - logp_old[mb])
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t[mb]
                policy_loss = -torch.min(ratio * adv_t[mb], clip_adv).mean()
                value_loss = nn.functional.mse_loss(value, ret_t[mb])
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy.mean()

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
