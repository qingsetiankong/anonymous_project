from __future__ import annotations

import os
import statistics
import time
from collections import deque
from pathlib import Path

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from algorithm.pasist_ppo import PasistPPO
from envs.pasist.pasist_robot import PasistLeggedRobot


class OnPolicyRunner:
    def __init__(
        self,
        env: PasistLeggedRobot,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cuda:0",
    ):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        actor_critic_class = self.cfg["policy_class_name"]
        from rl.pasist_actor_critic import PasistActorCritic

        num_actor_obs = env.num_obs
        num_critic_obs = env.num_privileged_obs

        actor_critic = PasistActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=env.num_actions,
            **self.policy_cfg,
        )
        actor_critic.to(self.device)

        alg_class = self.cfg["algorithm_class_name"]
        self.alg = PasistPPO(
            actor_critic=actor_critic,
            device=self.device,
            **self.alg_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[num_actor_obs],
            privileged_obs_shape=[num_critic_obs],
            action_shape=[env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0

        self.env.reset()

    def _aggregate_episode_metrics(self, ep_infos: list[dict]) -> dict[str, float]:
        if not ep_infos:
            return {}

        aggregated: dict[str, list[float]] = {}
        for ep_info in ep_infos:
            for key, value in ep_info.items():
                if isinstance(value, torch.Tensor):
                    tensor = value.detach()
                    if tensor.numel() == 0:
                        continue
                    aggregated.setdefault(key, []).append(float(tensor.float().mean().item()))
                else:
                    aggregated.setdefault(key, []).append(float(value))

        return {
            key: statistics.mean(values)
            for key, values in aggregated.items()
            if values
        }

    def _build_episode_log_lines(
        self,
        episode_metrics: dict[str, float],
        pad: int,
    ) -> list[str]:
        if not episode_metrics:
            return []

        preferred_keys = [
            "rew_tracking_lin_vel",
            "rew_tracking_ang_vel",
            "rew_orientation",
            "rew_action_rate",
            "rew_base_height",
            "rew_collision",
            "rew_termination",
            "max_command_x",
            "terrain_level",
        ]
        display_labels = {
            "rew_tracking_lin_vel": "Episode tracking_lin_vel",
            "rew_tracking_ang_vel": "Episode tracking_ang_vel",
            "rew_orientation": "Episode orientation",
            "rew_action_rate": "Episode action_rate",
            "rew_base_height": "Episode base_height",
            "rew_collision": "Episode collision",
            "rew_termination": "Episode termination",
            "max_command_x": "Episode max_command_x",
            "terrain_level": "Episode terrain_level",
        }

        ordered_keys: list[str] = [key for key in preferred_keys if key in episode_metrics]
        remaining_reward_keys = [
            key
            for key in sorted(episode_metrics)
            if key not in ordered_keys and key.startswith("rew_")
        ]
        ordered_keys.extend(remaining_reward_keys[:4])

        lines: list[str] = []
        for key in ordered_keys:
            label = display_labels.get(key, f"Episode {key}")
            lines.append(f"{label + ':':>{pad}} {episode_metrics[key]:.4f}")
        return lines

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = True,
    ):
        if self.log_dir is not None and self.writer is None:
            if SummaryWriter is not None:
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.randomize_episode_phases()

        obs_dict = self.env.get_observations()
        obs = obs_dict if isinstance(obs_dict, torch.Tensor) else obs_dict
        critic_obs = self.env.get_privileged_observations()
        if critic_obs is None:
            critic_obs = obs
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)

        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, 1, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, 1, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    ret = self.env.step(actions)
                    obs, privileged_obs, rewards, dones, infos = ret
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device).unsqueeze(-1)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.numel() > 0:
                            rewbuffer.extend(
                                cur_reward_sum[new_ids].reshape(-1).cpu().numpy().tolist()
                            )
                            lenbuffer.extend(
                                cur_episode_length[new_ids].reshape(-1).cpu().numpy().tolist()
                            )
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            start = stop
            # Clone tensors that were created under inference_mode before using
            # them outside the context manager.
            critic_obs = critic_obs.clone()
            self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_entropy = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self._log(
                    it=it,
                    num_learning_iterations=num_learning_iterations,
                    collection_time=collection_time,
                    learn_time=learn_time,
                    mean_value_loss=mean_value_loss,
                    mean_surrogate_loss=mean_surrogate_loss,
                    mean_entropy=mean_entropy,
                    ep_infos=ep_infos,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                )
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += collection_time + learn_time

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir, f"model_{self.current_learning_iteration}.pt"
            )
        )

    def _log(
        self,
        it: int,
        num_learning_iterations: int,
        collection_time: float,
        learn_time: float,
        mean_value_loss: float,
        mean_surrogate_loss: float,
        mean_entropy: float,
        ep_infos: list,
        rewbuffer: deque,
        lenbuffer: deque,
    ):
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (collection_time + learn_time)
        )
        mean_std = self.alg.actor_critic.actor.get_std().mean().item()
        episode_metrics = self._aggregate_episode_metrics(ep_infos)
        next_total_timesteps = (
            self.tot_timesteps + self.num_steps_per_env * self.env.num_envs
        )
        iteration_time = collection_time + learn_time
        next_total_time = self.tot_time + iteration_time
        completed_iterations = it + 1
        remaining_iterations = max(num_learning_iterations - completed_iterations, 0)
        eta_seconds = (
            next_total_time / completed_iterations * remaining_iterations
            if completed_iterations > 0
            else 0.0
        )

        if self.writer is not None:
            self.writer.add_scalar("Loss/value_function", mean_value_loss, it)
            self.writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
            self.writer.add_scalar("Loss/entropy", mean_entropy, it)
            self.writer.add_scalar(
                "Loss/learning_rate", self.alg.learning_rate, it
            )
            self.writer.add_scalar("Policy/mean_noise_std", mean_std, it)
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collection_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)

            if len(rewbuffer) > 0:
                self.writer.add_scalar(
                    "Train/mean_reward", statistics.mean(rewbuffer), it
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length", statistics.mean(lenbuffer), it
                )

            for key, value in episode_metrics.items():
                self.writer.add_scalar(f"Episode/{key}", value, it)

        if len(rewbuffer) > 0:
            mean_reward = statistics.mean(rewbuffer)
            mean_ep_len = statistics.mean(lenbuffer)
        else:
            mean_reward = 0.0
            mean_ep_len = 0.0

        episode_log_lines = self._build_episode_log_lines(
            episode_metrics=episode_metrics,
            pad=35,
        )
        log_string = (
            f"{'#' * 80}\n"
            f" Learning iteration {it}/{self.current_learning_iteration + num_learning_iterations} \n"
            f"{'Computation:':>35} {fps:.0f} steps/s "
            f"(collection: {collection_time:.3f}s, learning: {learn_time:.3f}s)\n"
            f"{'Value function loss:':>35} {mean_value_loss:.4f}\n"
            f"{'Surrogate loss:':>35} {mean_surrogate_loss:.4f}\n"
            f"{'Entropy:':>35} {mean_entropy:.4f}\n"
            f"{'Learning rate:':>35} {self.alg.learning_rate:.6f}\n"
            f"{'Mean action noise std:':>35} {mean_std:.2f}\n"
            f"{'Mean reward:':>35} {mean_reward:.2f}\n"
            f"{'Mean episode length:':>35} {mean_ep_len:.2f}\n"
        )
        if episode_log_lines:
            log_string += "\n".join(episode_log_lines) + "\n"
        log_string += (
            f"{'-' * 80}\n"
            f"{'Total timesteps:':>35} {next_total_timesteps}\n"
            f"{'Iteration time:':>35} {iteration_time:.2f}s\n"
            f"{'Total time:':>35} {next_total_time:.2f}s\n"
            f"{'ETA:':>35} {eta_seconds:.1f}s\n"
        )
        print(log_string)

    def save(self, path: str, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return loaded_dict.get("infos")
