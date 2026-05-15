from __future__ import annotations

from typing import Any

import torch


def _infer_device(*values: Any) -> torch.device:
    for value in values:
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_float_vector_or_batch(value: Any, device: torch.device) -> torch.Tensor:
    tensor = _as_float_tensor(value, device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor


def _as_reward_vector(value: Any, device: torch.device) -> torch.Tensor:
    tensor = _as_float_tensor(value, device=device)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        return tensor.squeeze(-1)
    return tensor


def _sum_except_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.sum().reshape(1)
    dims = tuple(range(1, tensor.ndim))
    return tensor.sum(dim=dims)


def _mean_except_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.mean().reshape(1)
    dims = tuple(range(1, tensor.ndim))
    return tensor.mean(dim=dims)


def _all_skill_ids_equal(skill_id: int | torch.Tensor | None, expected_skill_id: int) -> bool:
    if skill_id is None:
        return True
    device = _infer_device(skill_id)
    skill_tensor = _as_float_tensor(skill_id, device=device).to(dtype=torch.long)
    return bool(torch.all(skill_tensor == int(expected_skill_id)).item())


def pose_tracking_reward_torch(
    imitation_observation: Any,
    target_pose: Any,
    sigma: float = 1.0,
) -> torch.Tensor:
    device = _infer_device(imitation_observation, target_pose)
    observation = _as_float_vector_or_batch(imitation_observation, device=device)
    target = _as_float_vector_or_batch(target_pose, device=device)
    if observation.shape != target.shape:
        if observation.ndim >= 2 and target.ndim == 1 and observation.shape[-1] == target.shape[-1]:
            target = target.unsqueeze(0).expand_as(observation)
        else:
            raise ValueError("imitation_observation 和 target_pose 的维度必须一致，或满足 [B, D] 对 [D] 广播")

    error = torch.linalg.norm(observation - target, dim=-1 if observation.ndim >= 2 else 0)
    return torch.exp(-error / max(float(sigma), 1.0e-6))


def velocity_tracking_reward_torch(
    commanded_velocity: Any,
    measured_velocity: Any,
    sigma: float = 0.25,
) -> torch.Tensor:
    device = _infer_device(commanded_velocity, measured_velocity)
    commanded = _as_float_tensor(commanded_velocity, device=device)
    measured = _as_float_tensor(measured_velocity, device=device)
    return torch.exp(-torch.abs(commanded - measured) / max(float(sigma), 1.0e-6))


def velocity_tracking_xy_exp_reward_torch(
    commanded_linear_velocity_xy: Any,
    measured_linear_velocity_xy: Any,
    std: float = 0.5,
) -> torch.Tensor:
    device = _infer_device(commanded_linear_velocity_xy, measured_linear_velocity_xy)
    commanded = _as_float_vector_or_batch(commanded_linear_velocity_xy, device=device)
    measured = _as_float_vector_or_batch(measured_linear_velocity_xy, device=device)
    if commanded.shape != measured.shape:
        if measured.ndim >= 2 and commanded.ndim == 1 and measured.shape[-1] == commanded.shape[-1]:
            commanded = commanded.unsqueeze(0).expand_as(measured)
        else:
            raise ValueError("commanded_linear_velocity_xy 和 measured_linear_velocity_xy 的维度必须一致")
    error = torch.square(commanded - measured).sum(dim=-1 if measured.ndim >= 2 else 0)
    return torch.exp(-error / max(float(std) ** 2, 1.0e-6))


def yaw_tracking_exp_reward_torch(
    commanded_yaw_rate: Any,
    measured_yaw_rate: Any,
    std: float = 0.5,
) -> torch.Tensor:
    device = _infer_device(commanded_yaw_rate, measured_yaw_rate)
    commanded = _as_float_tensor(commanded_yaw_rate, device=device)
    measured = _as_float_tensor(measured_yaw_rate, device=device)
    error = torch.square(commanded - measured)
    return torch.exp(-error / max(float(std) ** 2, 1.0e-6))


def linear_velocity_z_l2_penalty_torch(measured_linear_velocity: Any) -> torch.Tensor:
    device = _infer_device(measured_linear_velocity)
    velocity = _as_float_vector_or_batch(measured_linear_velocity, device=device)
    if velocity.shape[-1] < 3:
        raise ValueError("measured_linear_velocity 至少需要包含 xyz 三个分量")
    return torch.square(velocity[..., 2])


def angular_velocity_xy_l2_penalty_torch(measured_angular_velocity: Any) -> torch.Tensor:
    device = _infer_device(measured_angular_velocity)
    angular_velocity = _as_float_vector_or_batch(measured_angular_velocity, device=device)
    if angular_velocity.shape[-1] < 2:
        raise ValueError("measured_angular_velocity 至少需要包含 x/y 两个分量")
    return torch.square(angular_velocity[..., :2]).sum(dim=-1 if angular_velocity.ndim >= 2 else 0)


def upright_posture_reward_torch(
    projected_gravity: Any,
    sigma: float = 0.25,
    upright_reference: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> torch.Tensor:
    device = _infer_device(projected_gravity)
    gravity = _as_float_vector_or_batch(projected_gravity, device=device)
    reference = torch.tensor(upright_reference, dtype=torch.float32, device=device)
    error = torch.linalg.norm(gravity - reference, dim=-1 if gravity.ndim >= 2 else 0)
    return torch.exp(-error / max(float(sigma), 1.0e-6))


def flat_orientation_l2_penalty_torch(projected_gravity: Any) -> torch.Tensor:
    device = _infer_device(projected_gravity)
    gravity = _as_float_vector_or_batch(projected_gravity, device=device)
    return torch.square(gravity[..., :2]).sum(dim=-1 if gravity.ndim >= 2 else 0)


def base_height_reward_torch(
    base_height: Any,
    target_base_height: Any,
    sigma: float = 0.05,
) -> torch.Tensor:
    device = _infer_device(base_height, target_base_height)
    height = _as_float_tensor(base_height, device=device)
    target = _as_float_tensor(target_base_height, device=device)
    return torch.exp(-torch.abs(height - target) / max(float(sigma), 1.0e-6))


def command_consistency_reward_torch(
    skill_id: Any,
    command_skill_id: Any,
) -> torch.Tensor:
    device = _infer_device(skill_id, command_skill_id)
    current_skill = _as_float_tensor(skill_id, device=device).to(dtype=torch.long)
    command_skill = _as_float_tensor(command_skill_id, device=device).to(dtype=torch.long)
    return (current_skill == command_skill).to(dtype=torch.float32)


def compute_walk_task_reward_torch(
    imitation_observation: Any,
    commanded_velocity: Any,
    measured_velocity: Any,
    measured_linear_velocity: Any | None = None,
    measured_angular_velocity: Any | None = None,
    projected_gravity: Any | None = None,
    posture_weight: float = 0.5,
    velocity_weight: float = 1.0,
    posture_sigma: float = 0.25,
    velocity_sigma: float = 0.25,
    commanded_yaw_rate: Any | None = None,
    measured_yaw_rate: Any | None = None,
    yaw_weight: float = 0.0,
    yaw_sigma: float = 0.25,
    lin_vel_z_weight: float = 0.0,
    ang_vel_xy_weight: float = 0.0,
    flat_orientation_weight: float = 0.0,
    base_height: Any | None = None,
    target_base_height: Any | None = None,
    height_weight: float = 0.0,
    height_sigma: float = 0.05,
) -> torch.Tensor:
    device = _infer_device(
        imitation_observation,
        commanded_velocity,
        measured_velocity,
        measured_linear_velocity,
        measured_angular_velocity,
        projected_gravity,
        base_height,
        target_base_height,
    )
    imitation_tensor = _as_float_vector_or_batch(imitation_observation, device=device)
    commanded_velocity_tensor = _as_float_tensor(commanded_velocity, device=device)

    if measured_linear_velocity is None:
        measured_velocity_tensor = _as_float_tensor(measured_velocity, device=device)
        measured_linear_velocity_xy = torch.stack(
            [measured_velocity_tensor, torch.zeros_like(measured_velocity_tensor)],
            dim=-1,
        )
    else:
        linear_velocity = _as_float_vector_or_batch(measured_linear_velocity, device=device)
        if linear_velocity.shape[-1] < 2:
            raise ValueError("measured_linear_velocity 至少需要包含 x/y 两个分量")
        measured_linear_velocity_xy = linear_velocity[..., :2]

    commanded_linear_velocity_xy = torch.stack(
        [commanded_velocity_tensor, torch.zeros_like(commanded_velocity_tensor)],
        dim=-1,
    )
    reward = (
        float(velocity_weight)
        * velocity_tracking_xy_exp_reward_torch(
            commanded_linear_velocity_xy=commanded_linear_velocity_xy,
            measured_linear_velocity_xy=measured_linear_velocity_xy,
            std=velocity_sigma,
        )
    )

    if yaw_weight != 0.0 and commanded_yaw_rate is not None and measured_yaw_rate is not None:
        reward = reward + float(yaw_weight) * yaw_tracking_exp_reward_torch(
            commanded_yaw_rate=commanded_yaw_rate,
            measured_yaw_rate=measured_yaw_rate,
            std=yaw_sigma,
        )

    if lin_vel_z_weight != 0.0 and measured_linear_velocity is not None:
        reward = reward + float(lin_vel_z_weight) * linear_velocity_z_l2_penalty_torch(measured_linear_velocity)

    if ang_vel_xy_weight != 0.0 and measured_angular_velocity is not None:
        reward = reward + float(ang_vel_xy_weight) * angular_velocity_xy_l2_penalty_torch(measured_angular_velocity)

    if flat_orientation_weight != 0.0 and projected_gravity is not None:
        reward = reward + float(flat_orientation_weight) * flat_orientation_l2_penalty_torch(projected_gravity)

    if posture_weight != 0.0 and projected_gravity is not None:
        reward = reward + float(posture_weight) * upright_posture_reward_torch(
            projected_gravity=projected_gravity,
            sigma=posture_sigma,
        )

    if height_weight != 0.0 and base_height is not None and target_base_height is not None:
        reward = reward + float(height_weight) * base_height_reward_torch(
            base_height=base_height,
            target_base_height=target_base_height,
            sigma=height_sigma,
        )

    if reward.ndim == 0:
        return reward.reshape(1)
    return reward


def compute_task_reward_torch(
    imitation_observation: Any,
    target_pose: Any,
    commanded_velocity: Any | None = None,
    measured_velocity: Any | None = None,
    measured_linear_velocity: Any | None = None,
    measured_angular_velocity: Any | None = None,
    projected_gravity: Any | None = None,
    skill_id: int | torch.Tensor | None = None,
    command_skill_id: int | torch.Tensor | None = None,
    pose_weight: float = 1.0,
    velocity_weight: float = 0.0,
    command_weight: float = 0.0,
    pose_sigma: float = 1.0,
    velocity_sigma: float = 0.25,
    commanded_yaw_rate: Any | None = None,
    measured_yaw_rate: Any | None = None,
    yaw_weight: float = 0.0,
    yaw_sigma: float = 0.25,
    lin_vel_z_weight: float = 0.0,
    ang_vel_xy_weight: float = 0.0,
    flat_orientation_weight: float = 0.0,
    base_height: Any | None = None,
    target_base_height: Any | None = None,
    height_weight: float = 0.0,
    height_sigma: float = 0.05,
) -> torch.Tensor:
    device = _infer_device(
        imitation_observation,
        target_pose,
        measured_velocity,
        measured_linear_velocity,
        measured_angular_velocity,
        projected_gravity,
        base_height,
        target_base_height,
    )
    if _all_skill_ids_equal(skill_id, expected_skill_id=0):
        return compute_walk_task_reward_torch(
            imitation_observation=imitation_observation,
            commanded_velocity=0.0 if commanded_velocity is None else commanded_velocity,
            measured_velocity=0.0 if measured_velocity is None else measured_velocity,
            measured_linear_velocity=measured_linear_velocity,
            measured_angular_velocity=measured_angular_velocity,
            projected_gravity=projected_gravity,
            posture_weight=pose_weight,
            velocity_weight=velocity_weight,
            posture_sigma=pose_sigma,
            velocity_sigma=velocity_sigma,
            commanded_yaw_rate=commanded_yaw_rate,
            measured_yaw_rate=measured_yaw_rate,
            yaw_weight=yaw_weight,
            yaw_sigma=yaw_sigma,
            lin_vel_z_weight=lin_vel_z_weight,
            ang_vel_xy_weight=ang_vel_xy_weight,
            flat_orientation_weight=flat_orientation_weight,
            base_height=base_height,
            target_base_height=target_base_height,
            height_weight=height_weight,
            height_sigma=height_sigma,
        )

    reward = float(pose_weight) * pose_tracking_reward_torch(
        imitation_observation=imitation_observation,
        target_pose=target_pose,
        sigma=pose_sigma,
    )
    if commanded_velocity is not None and measured_velocity is not None and velocity_weight != 0.0:
        reward = reward + float(velocity_weight) * velocity_tracking_reward_torch(
            commanded_velocity=commanded_velocity,
            measured_velocity=measured_velocity,
            sigma=velocity_sigma,
        )
    if skill_id is not None and command_skill_id is not None and command_weight != 0.0:
        reward = reward + float(command_weight) * command_consistency_reward_torch(
            skill_id=skill_id,
            command_skill_id=command_skill_id,
        )
    return _as_reward_vector(reward, device=device)


def compute_regularization_reward_torch(
    action: Any,
    previous_action: Any | None = None,
    observation: Any | None = None,
    reference_observation: Any | None = None,
    action_weight: float = 1.0e-3,
    smoothness_weight: float = 1.0e-3,
    stability_weight: float = 0.0,
    joint_velocity: Any | None = None,
    previous_joint_velocity: Any | None = None,
    joint_acceleration_weight: float = 0.0,
    base_angular_velocity: Any | None = None,
    roll_pitch_rate_weight: float = 0.0,
    yaw_rate_weight: float = 0.0,
    lateral_velocity: Any | None = None,
    lateral_velocity_weight: float = 0.0,
) -> torch.Tensor:
    device = _infer_device(
        action,
        previous_action,
        observation,
        reference_observation,
        joint_velocity,
        previous_joint_velocity,
        base_angular_velocity,
        lateral_velocity,
    )
    action_tensor = _as_float_vector_or_batch(action, device=device)
    reward = -float(action_weight) * _sum_except_batch(torch.square(action_tensor))

    if previous_action is not None:
        previous_action_tensor = _as_float_vector_or_batch(previous_action, device=device)
        reward = reward - float(smoothness_weight) * _sum_except_batch(torch.square(action_tensor - previous_action_tensor))

    if reference_observation is not None and stability_weight != 0.0 and observation is not None:
        current_observation = _as_float_vector_or_batch(observation, device=device)
        reference_tensor = _as_float_vector_or_batch(reference_observation, device=device)
        reward = reward - float(stability_weight) * _mean_except_batch(torch.square(current_observation - reference_tensor))

    if joint_velocity is not None and previous_joint_velocity is not None and joint_acceleration_weight != 0.0:
        current_joint_velocity = _as_float_vector_or_batch(joint_velocity, device=device)
        previous_joint_velocity_tensor = _as_float_vector_or_batch(previous_joint_velocity, device=device)
        reward = reward - float(joint_acceleration_weight) * _sum_except_batch(
            torch.square(current_joint_velocity - previous_joint_velocity_tensor)
        )

    if base_angular_velocity is not None and roll_pitch_rate_weight != 0.0:
        angular_velocity = _as_float_vector_or_batch(base_angular_velocity, device=device)
        reward = reward - float(roll_pitch_rate_weight) * _sum_except_batch(torch.square(angular_velocity[..., :2]))

    if base_angular_velocity is not None and yaw_rate_weight != 0.0:
        angular_velocity = _as_float_vector_or_batch(base_angular_velocity, device=device)
        reward = reward - float(yaw_rate_weight) * _sum_except_batch(torch.square(angular_velocity[..., 2:3]))

    if lateral_velocity is not None and lateral_velocity_weight != 0.0:
        lateral_velocity_tensor = _as_float_tensor(lateral_velocity, device=device)
        reward = reward - float(lateral_velocity_weight) * torch.square(lateral_velocity_tensor)

    return _as_reward_vector(reward, device=device)


def compute_task_weight_torch(task_reward: Any, sigma_t: float) -> torch.Tensor:
    device = _infer_device(task_reward)
    task_reward_tensor = _as_reward_vector(task_reward, device=device)
    return torch.exp(-torch.abs(task_reward_tensor - float(sigma_t)))


def compute_total_reward_torch(
    task_reward: Any,
    sil_reward: Any,
    regularization_reward: Any,
    omega_t: Any,
    omega_sil: Any,
    omega_r: float = 1.0,
) -> torch.Tensor:
    device = _infer_device(task_reward, sil_reward, regularization_reward, omega_t, omega_sil)
    task_reward_tensor = _as_reward_vector(task_reward, device=device)
    sil_reward_tensor = _as_reward_vector(sil_reward, device=device)
    regularization_reward_tensor = _as_reward_vector(regularization_reward, device=device)
    omega_t_tensor = _as_reward_vector(omega_t, device=device)
    omega_sil_tensor = _as_reward_vector(omega_sil, device=device)
    return (
        omega_sil_tensor * omega_t_tensor * sil_reward_tensor
        + (1.0 - omega_t_tensor) * task_reward_tensor
        + float(omega_r) * regularization_reward_tensor
    )


def compute_reward_terms_torch(
    task_reward: Any,
    sil_reward: Any,
    regularization_reward: Any,
    sigma_t: float,
    omega_sil: Any,
    omega_r: float = 1.0,
) -> dict[str, torch.Tensor]:
    device = _infer_device(task_reward, sil_reward, regularization_reward, omega_sil)
    task_reward_tensor = _as_reward_vector(task_reward, device=device)
    sil_reward_tensor = _as_reward_vector(sil_reward, device=device)
    regularization_reward_tensor = _as_reward_vector(regularization_reward, device=device)
    omega_sil_tensor = _as_reward_vector(omega_sil, device=device)
    omega_t_tensor = compute_task_weight_torch(task_reward=task_reward_tensor, sigma_t=sigma_t)
    total_reward_tensor = compute_total_reward_torch(
        task_reward=task_reward_tensor,
        sil_reward=sil_reward_tensor,
        regularization_reward=regularization_reward_tensor,
        omega_t=omega_t_tensor,
        omega_sil=omega_sil_tensor,
        omega_r=omega_r,
    )
    return {
        "task_reward": task_reward_tensor,
        "sil_reward": sil_reward_tensor,
        "regularization_reward": regularization_reward_tensor,
        "omega_t": omega_t_tensor,
        "omega_sil": omega_sil_tensor,
        "omega_r": torch.full_like(task_reward_tensor, float(omega_r)),
        "total_reward": total_reward_tensor,
    }
