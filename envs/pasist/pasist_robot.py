from __future__ import annotations

import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

# IsaacGym tensor helpers (must match legged_gym usage)
from isaacgym.torch_utils import quat_rotate_inverse
from isaacgym import gymtorch
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor


class PasistLeggedRobot(LeggedRobot):
    """
    PASIST locomotion env built on legged_gym's LeggedRobot.

    Produces 49-dim policy observations and 52-dim critic observations
    (matching the original Pasist dimensions).

    Uses legged_gym's reward system, termination, and curriculum.
    """

    imitation_obs_dim = 14

    def __init__(
        self,
        cfg: LeggedRobotCfg,
        sim_params,
        physics_engine,
        sim_device,
        headless,
        obs_history_len: int = 0,
        target_pose_bank: dict[int, np.ndarray] | None = None,
    ):
        self.obs_history_len = int(obs_history_len)
        self._target_pose_bank = target_pose_bank or {}
        self._initial_target_pose_bank = dict(self._target_pose_bank)

        # Policy obs: 45 proprio + 4 command_onehot = 49
        # Critic obs: 48 proprio + 4 command_onehot = 52
        self.policy_proprio_dim = 45
        self.critic_proprio_dim = 48
        self.command_onehot_dim = 4
        self.policy_obs_dim = self.policy_proprio_dim + self.command_onehot_dim
        self.critic_obs_dim = self.critic_proprio_dim + self.command_onehot_dim

        cfg.env.num_observations = self.policy_obs_dim
        cfg.env.num_privileged_obs = self.critic_obs_dim
        cfg.env.num_actions = 12

        # legged_gym LeggedRobot expects these to be set
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        if self.obs_history_len > 0:
            self.obs_history_buf = torch.zeros(
                self.num_envs,
                self.obs_history_len,
                self.policy_proprio_dim,
                dtype=torch.float,
                device=self.device,
            )
        else:
            self.obs_history_buf = None

        self.imitation_obs_buf = torch.zeros(
            self.num_envs, self.imitation_obs_dim, dtype=torch.float, device=self.device
        )

        # Override the noise scale vec to match our observation layout
        self.noise_scale_vec = self._make_noise_scale_vec()

    def _make_noise_scale_vec(self) -> torch.Tensor:
        noise_vec = torch.zeros(self.policy_proprio_dim, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # Layout: base_ang_vel(3), projected_gravity(3), commands(3), dof_pos(12), dof_vel(12), last_action(12)
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.0
        return noise_vec

    def _command_onehot(self) -> torch.Tensor:
        """Walk skill one-hot: [1, 0, 0, 0]."""
        onehot = torch.zeros(self.num_envs, self.command_onehot_dim,
                             dtype=torch.float, device=self.device)
        onehot[:, 0] = 1.0
        return onehot

    def compute_observations(self):
        policy_proprio = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        if self.add_noise:
            policy_proprio += (
                2 * torch.rand_like(policy_proprio) - 1
            ) * self.noise_scale_vec

        clip_obs = self.cfg.normalization.clip_observations
        policy_proprio = torch.clip(policy_proprio, -clip_obs, clip_obs)

        cmd_onehot = self._command_onehot()

        if self.obs_history_len > 0 and self.obs_history_buf is not None:
            hist_part = self.obs_history_buf.view(self.num_envs, -1)
            self.obs_buf = torch.cat([policy_proprio, cmd_onehot, hist_part], dim=-1)

            init_flag = (self.episode_length_buf <= 1)[:, None, None]
            self.obs_history_buf = torch.where(
                init_flag,
                policy_proprio.unsqueeze(1).repeat(1, self.obs_history_len, 1),
                torch.cat(
                    [self.obs_history_buf[:, 1:], policy_proprio.unsqueeze(1)], dim=1
                ),
            )
        else:
            self.obs_buf = torch.cat([policy_proprio, cmd_onehot], dim=-1)

        # Critic observations: base_lin_vel instead of base_ang_vel
        critic_proprio = torch.cat(
            (
                self.base_lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        critic_proprio = torch.clip(critic_proprio, -clip_obs, clip_obs)
        self.privileged_obs_buf = torch.cat([critic_proprio, cmd_onehot], dim=-1)

        self._compute_imitation_obs()

    def _compute_imitation_obs(self):
        base_height = self.root_states[:, 2:3]
        joint_pos_rel = self.dof_pos - self.default_dof_pos
        skill_id = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.imitation_obs_buf = torch.cat(
            [base_height, joint_pos_rel, skill_id], dim=-1
        )

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def _post_physics_step_callback(self):
        env_ids = (
            self.episode_length_buf
            % int(self.cfg.commands.resampling_time / self.dt)
            == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        self.reset_buf |= torch.logical_or(
            torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.cfg.commands.curriculum and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self.update_command_curriculum(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        if self.obs_history_buf is not None:
            self.obs_history_buf[env_ids] = 0.0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids])
                / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges[
                "lin_vel_x"
            ][1]

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def get_imitation_obs(self) -> torch.Tensor:
        return self.imitation_obs_buf

    def get_target_pose(self, skill_id: int = 0) -> np.ndarray:
        if skill_id in self._target_pose_bank:
            return self._target_pose_bank[skill_id]
        return self.imitation_obs_buf[0].detach().cpu().numpy().astype(np.float32)

    def randomize_episode_phases(self) -> None:
        buf = getattr(self, "episode_length_buf", None)
        max_len = getattr(self, "max_episode_length", None)
        if buf is None or max_len is None:
            return
        try:
            max_steps = int(max_len)
        except (TypeError, ValueError):
            return
        if max_steps <= 1:
            return
        buf.copy_(
            torch.randint(
                0, max_steps, size=buf.shape, device=buf.device, dtype=buf.dtype,
            )
        )

    # ---- reward functions (legged_gym naming convention) ----
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(
            torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1
        )

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.contact_forces[:, self.penalised_contact_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum(
            (
                torch.abs(self.dof_vel)
                - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_torque_limits(self):
        return torch.sum(
            (
                torch.abs(self.torques)
                - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_stand_still(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
