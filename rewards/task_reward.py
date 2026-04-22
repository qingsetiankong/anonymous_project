from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray:
    """
    将输入统一转成 `float32` numpy 数组。

    这里不强制展平成一维，原因是 task reward 后面需要同时支持：
    - 单样本输入：shape = [D]
    - 并行环境输入：shape = [B, D]

    参数:
    - value:
      任意可被 `np.asarray(..., dtype=np.float32)` 接受的对象，
      例如标量、list、numpy 数组等。

    返回:
    - `np.ndarray`
      dtype 为 `float32`，shape 尽量保持输入原状。
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _as_float_vector_or_batch(value: Any) -> np.ndarray:
    """
    将输入整理成“最后一维是特征维”的数组。

    约定:
    - 标量会被整理成 shape = [1]
    - 一维向量保持 shape = [D]
    - 二维及以上张量保持原 shape，不打乱 batch 维

    这个辅助函数主要给 `pose_tracking_reward()` 使用。
    """
    array = _as_float_array(value)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def _maybe_scalar(value: np.ndarray | np.generic | float) -> float | np.ndarray:
    """
    将 0 维 numpy 结果转回 Python float。

    这样函数在单样本输入时返回 `float`，
    在 batched 输入时返回 `np.ndarray`，接口更直观。
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.astype(np.float32, copy=False)


def _all_skill_ids_equal(skill_id: int | np.ndarray | None, expected_skill_id: int) -> bool:
    """
    判断当前 batch 是否全部属于同一个 skill。

    这有助于在单技能阶段走更清晰的 reward 分支。
    """
    if skill_id is None:
        return True
    skill_array = np.asarray(skill_id)
    return bool(np.all(skill_array == int(expected_skill_id)))


def pose_tracking_reward(
    imitation_observation: Any,
    target_pose: Any,
    sigma: float = 1.0,
) -> float | np.ndarray:
    """
    计算最基础的姿态跟踪奖励。

    设计思路:
    - imitation_observation 越接近 target_pose，奖励越高
    - 用指数函数把距离映射到 (0, 1]，便于和论文中的其它奖励项组合

    参数:
    - imitation_observation:
      当前时刻用于模仿的观测子空间。
      支持两种常见形状：
      - 单样本：shape = [D]
      - batched：shape = [B, D]
    - target_pose:
      当前 skill 对应的目标姿态。
      支持：
      - 单个目标姿态：shape = [D]
      - 与输入逐样本对应的 batched 目标：shape = [B, D]
    - sigma:
      距离缩放超参数，必须为正数。

    返回:
    - 如果输入是单样本，返回 `float`
    - 如果输入是 batched，返回 shape = [B] 的 `np.ndarray`

    值域:
    - 理论上在 `(0, 1]`
    """
    observation = _as_float_vector_or_batch(imitation_observation)
    target = _as_float_vector_or_batch(target_pose)
    if observation.shape != target.shape:
        # 允许 target_pose 是单帧姿态 [D]，自动广播到 batched 输入 [B, D]。
        if observation.ndim >= 2 and target.ndim == 1 and observation.shape[-1] == target.shape[-1]:
            target = np.broadcast_to(target, observation.shape)
        else:
            raise ValueError("imitation_observation 和 target_pose 的维度必须一致，或满足 [B, D] 对 [D] 广播")

    error = np.linalg.norm(observation - target, axis=-1 if observation.ndim >= 2 else 0)
    reward = np.exp(-error / max(float(sigma), 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def upright_posture_reward(
    imitation_observation: Any | None = None,
    sigma: float = 0.25,
    projected_gravity_slice: tuple[int, int] = (3, 6),
    upright_reference: tuple[float, float, float] = (0.0, 0.0, -1.0),
    projected_gravity: Any | None = None,
) -> float | np.ndarray:
    """
    计算机身直立奖励。

    这是当前 `walk` 技能里更贴近 PASIST 风格的 task reward 子项：
    - 它不直接让 `r_T` 去拟合整帧 target pose
    - 而是鼓励机器人保持稳定、自然的直立姿态

    参数:
    - `imitation_observation`:
      当前 imitation 特征，默认假设其中 `[3:6]` 是 projected_gravity
    - `sigma`:
      奖励缩放系数
    - `projected_gravity_slice`:
      projected_gravity 在 imitation 向量中的切片
    - `upright_reference`:
      理想直立时的 projected_gravity，当前假设为 `(0, 0, -1)`

    返回:
    - 单样本时为 `float`
    - batched 时为 shape = [B] 的 `np.ndarray`
    """
    if projected_gravity is None:
        if imitation_observation is None:
            raise ValueError("upright_posture_reward 需要 imitation_observation 或 projected_gravity 之一")
        observation = _as_float_vector_or_batch(imitation_observation)
        start, end = projected_gravity_slice
        if observation.shape[-1] < end:
            raise ValueError(
                "imitation_observation 维度不足，无法提取 projected_gravity；"
                f"需要至少到索引 {end}，实际为 {observation.shape[-1]}"
            )
        projected_gravity = observation[..., start:end]
    else:
        projected_gravity = _as_float_vector_or_batch(projected_gravity)

    reference = np.asarray(upright_reference, dtype=np.float32)
    error = np.linalg.norm(projected_gravity - reference, axis=-1 if projected_gravity.ndim >= 2 else 0)
    reward = np.exp(-error / max(float(sigma), 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def velocity_tracking_reward(
    commanded_velocity: float | np.ndarray,
    measured_velocity: float | np.ndarray,
    sigma: float = 0.25,
) -> float | np.ndarray:
    """
    计算速度跟踪奖励。

    这个奖励项适合与你的 skill command 中的 `velocity` 分量联动，
    让策略不仅学会姿态，还学会以命令要求的速度执行该技能。

    输入:
    - commanded_velocity:
      期望速度，可以是单个标量，也可以是 shape = [B] 的数组。
    - measured_velocity:
      实际测得速度，可以是单个标量，也可以是 shape = [B] 的数组。
    - sigma:
      距离缩放超参数，必须为正数。

    输出:
    - 单样本时返回 `float`
    - batched 时返回 shape = [B] 的 `np.ndarray`
    """
    commanded = _as_float_array(commanded_velocity)
    measured = _as_float_array(measured_velocity)
    error = np.abs(commanded - measured)
    reward = np.exp(-error / max(float(sigma), 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def velocity_tracking_xy_exp_reward(
    commanded_linear_velocity_xy: Any,
    measured_linear_velocity_xy: Any,
    std: float = 0.5,
) -> float | np.ndarray:
    """
    复刻 IsaacLab `track_lin_vel_xy_exp` 的指数核速度跟踪奖励。

    公式:
    - `exp(-||v_cmd_xy - v_meas_xy||^2 / std^2)`
    """
    commanded = _as_float_vector_or_batch(commanded_linear_velocity_xy)
    measured = _as_float_vector_or_batch(measured_linear_velocity_xy)
    if commanded.shape != measured.shape:
        if measured.ndim >= 2 and commanded.ndim == 1 and measured.shape[-1] == commanded.shape[-1]:
            commanded = np.broadcast_to(commanded, measured.shape)
        else:
            raise ValueError("commanded_linear_velocity_xy 和 measured_linear_velocity_xy 的维度必须一致")

    error = np.sum(np.square(commanded - measured), axis=-1 if measured.ndim >= 2 else 0)
    reward = np.exp(-error / max(float(std) ** 2, 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def yaw_tracking_exp_reward(
    commanded_yaw_rate: float | np.ndarray,
    measured_yaw_rate: float | np.ndarray,
    std: float = 0.5,
) -> float | np.ndarray:
    """
    复刻 IsaacLab `track_ang_vel_z_exp` 的指数核 yaw 跟踪奖励。
    """
    commanded = _as_float_array(commanded_yaw_rate)
    measured = _as_float_array(measured_yaw_rate)
    error = np.square(commanded - measured)
    reward = np.exp(-error / max(float(std) ** 2, 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def linear_velocity_z_l2_penalty(measured_linear_velocity: Any) -> float | np.ndarray:
    """
    复刻 IsaacLab `lin_vel_z_l2` 的竖直速度惩罚。
    """
    velocity = _as_float_vector_or_batch(measured_linear_velocity)
    if velocity.shape[-1] < 3:
        raise ValueError("measured_linear_velocity 至少需要包含 xyz 三个分量")
    penalty = np.square(velocity[..., 2])
    return _maybe_scalar(penalty)


def angular_velocity_xy_l2_penalty(measured_angular_velocity: Any) -> float | np.ndarray:
    """
    复刻 IsaacLab `ang_vel_xy_l2` 的 roll / pitch 角速度惩罚。
    """
    angular_velocity = _as_float_vector_or_batch(measured_angular_velocity)
    if angular_velocity.shape[-1] < 2:
        raise ValueError("measured_angular_velocity 至少需要包含 x/y 两个分量")
    penalty = np.sum(np.square(angular_velocity[..., :2]), axis=-1 if angular_velocity.ndim >= 2 else 0)
    return _maybe_scalar(penalty)


def flat_orientation_l2_penalty(
    imitation_observation: Any | None = None,
    projected_gravity_slice: tuple[int, int] = (3, 6),
    projected_gravity: Any | None = None,
) -> float | np.ndarray:
    """
    复刻 IsaacLab `flat_orientation_l2` 的机身水平姿态惩罚。

    当前 imitation observation 中 `[3:6]` 对应 projected_gravity。
    该项直接惩罚其 x/y 分量平方和。
    """
    if projected_gravity is None:
        if imitation_observation is None:
            raise ValueError("flat_orientation_l2_penalty 需要 imitation_observation 或 projected_gravity 之一")
        observation = _as_float_vector_or_batch(imitation_observation)
        start, end = projected_gravity_slice
        if observation.shape[-1] < end:
            raise ValueError(
                "imitation_observation 维度不足，无法提取 projected_gravity；"
                f"需要至少到索引 {end}，实际为 {observation.shape[-1]}"
            )
        projected_gravity = observation[..., start:end]
    else:
        projected_gravity = _as_float_vector_or_batch(projected_gravity)

    penalty = np.sum(np.square(projected_gravity[..., :2]), axis=-1 if projected_gravity.ndim >= 2 else 0)
    return _maybe_scalar(penalty)


def yaw_tracking_reward(
    commanded_yaw_rate: float | np.ndarray,
    measured_yaw_rate: float | np.ndarray,
    sigma: float = 0.25,
) -> float | np.ndarray:
    """
    计算 yaw 角速度跟踪奖励。

    当前训练入口还没有真正传入 yaw command，
    但这里先把接口留好，后续只要 trainer / env 提供对应量即可启用。
    """
    commanded = _as_float_array(commanded_yaw_rate)
    measured = _as_float_array(measured_yaw_rate)
    error = np.abs(commanded - measured)
    reward = np.exp(-error / max(float(sigma), 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def base_height_reward(
    base_height: float | np.ndarray,
    target_base_height: float | np.ndarray,
    sigma: float = 0.05,
) -> float | np.ndarray:
    """
    计算 base height 跟踪奖励。

    PASIST 原文里 `walk / crawl / stilt` 的一个关键区分因素就是 base height。
    当前最小环境还没有稳定把 base height 传进 trainer，因此这里只预留接口。
    """
    height = _as_float_array(base_height)
    target = _as_float_array(target_base_height)
    error = np.abs(height - target)
    reward = np.exp(-error / max(float(sigma), 1e-6)).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def command_consistency_reward(
    skill_id: int | np.ndarray,
    command_skill_id: int | np.ndarray,
) -> float | np.ndarray:
    """
    一个简单的命令一致性奖励。

    如果当前样本所属的 skill 与 command 指定的 skill 一致，给 1.0；
    否则给 0.0。

    这个接口主要是为了在 mock 环境或早期复现阶段保留一个最小可用的
    skill-conditioned reward 信号。

    输入:
    - `skill_id` / `command_skill_id`:
      可以是单个整数，也可以是 shape = [B] 的整数数组

    输出:
    - 单样本时返回 `float`
    - batched 时返回 shape = [B] 的 `np.ndarray`
    """
    current_skill = np.asarray(skill_id)
    command_skill = np.asarray(command_skill_id)
    reward = (current_skill == command_skill).astype(np.float32, copy=False)
    return _maybe_scalar(reward)


def compute_walk_task_reward(
    imitation_observation: Any,
    commanded_velocity: float | np.ndarray,
    measured_velocity: float | np.ndarray,
    measured_linear_velocity: Any | None = None,
    measured_angular_velocity: Any | None = None,
    projected_gravity: Any | None = None,
    posture_weight: float = 0.5,
    velocity_weight: float = 1.0,
    posture_sigma: float = 0.25,
    velocity_sigma: float = 0.25,
    commanded_yaw_rate: float | np.ndarray | None = None,
    measured_yaw_rate: float | np.ndarray | None = None,
    yaw_weight: float = 0.0,
    yaw_sigma: float = 0.25,
    lin_vel_z_weight: float = 0.0,
    ang_vel_xy_weight: float = 0.0,
    flat_orientation_weight: float = 0.0,
    base_height: float | np.ndarray | None = None,
    target_base_height: float | np.ndarray | None = None,
    height_weight: float = 0.0,
    height_sigma: float = 0.05,
) -> float | np.ndarray:
    """
    计算当前单技能 `walk` 的 task reward。

    设计原则:
    - 保持 PASIST 的“trainer 自己重算 r_T”结构
    - 但对 `walk` 采用更贴近 IsaacLab velocity locomotion 的项
    - 把 `target_pose` 继续留给 DTW / SIL，不塞进 walk 的主任务奖励

    当前优先采用的官方同款项:
    - `track_lin_vel_xy_exp`
    - `track_ang_vel_z_exp`
    - `lin_vel_z_l2`
    - `ang_vel_xy_l2`
    - `flat_orientation_l2`
    """
    if measured_linear_velocity is None:
        measured_linear_velocity_xy = np.stack(
            [
                _as_float_array(measured_velocity),
                np.zeros_like(_as_float_array(measured_velocity), dtype=np.float32),
            ],
            axis=-1,
        )
    else:
        linear_velocity = _as_float_vector_or_batch(measured_linear_velocity)
        if linear_velocity.shape[-1] < 2:
            raise ValueError("measured_linear_velocity 至少需要包含 x/y 两个分量")
        measured_linear_velocity_xy = linear_velocity[..., :2]

    commanded_velocity_array = _as_float_array(commanded_velocity)
    commanded_linear_velocity_xy = np.stack(
        [
            commanded_velocity_array,
            np.zeros_like(commanded_velocity_array, dtype=np.float32),
        ],
        axis=-1,
    )

    reward = np.asarray(
        float(velocity_weight)
        * velocity_tracking_xy_exp_reward(
            commanded_linear_velocity_xy=commanded_linear_velocity_xy,
            measured_linear_velocity_xy=measured_linear_velocity_xy,
            std=velocity_sigma,
        ),
        dtype=np.float32,
    )

    if (
        yaw_weight != 0.0
        and commanded_yaw_rate is not None
        and measured_yaw_rate is not None
    ):
        reward = reward + np.asarray(
            float(yaw_weight)
            * yaw_tracking_exp_reward(
                commanded_yaw_rate=commanded_yaw_rate,
                measured_yaw_rate=measured_yaw_rate,
                std=yaw_sigma,
            ),
            dtype=np.float32,
        )

    if lin_vel_z_weight != 0.0 and measured_linear_velocity is not None:
        reward = reward + np.asarray(
            float(lin_vel_z_weight)
            * linear_velocity_z_l2_penalty(
                measured_linear_velocity=measured_linear_velocity,
            ),
            dtype=np.float32,
        )

    if ang_vel_xy_weight != 0.0 and measured_angular_velocity is not None:
        reward = reward + np.asarray(
            float(ang_vel_xy_weight)
            * angular_velocity_xy_l2_penalty(
                measured_angular_velocity=measured_angular_velocity,
            ),
            dtype=np.float32,
        )

    if flat_orientation_weight != 0.0:
        reward = reward + np.asarray(
            float(flat_orientation_weight)
            * flat_orientation_l2_penalty(
                imitation_observation=imitation_observation,
                projected_gravity=projected_gravity,
            ),
            dtype=np.float32,
        )

    # 为了不破坏旧实验，保留原来的直立奖励接口。
    if posture_weight != 0.0:
        reward = reward + np.asarray(
            float(posture_weight)
            * upright_posture_reward(
                imitation_observation=imitation_observation,
                sigma=posture_sigma,
                projected_gravity=projected_gravity,
            ),
            dtype=np.float32,
        )

    if (
        height_weight != 0.0
        and base_height is not None
        and target_base_height is not None
    ):
        reward = reward + np.asarray(
            float(height_weight)
            * base_height_reward(
                base_height=base_height,
                target_base_height=target_base_height,
                sigma=height_sigma,
            ),
            dtype=np.float32,
        )

    return _maybe_scalar(reward)


def compute_task_reward(
    imitation_observation: Any,
    target_pose: Any,
    commanded_velocity: float | np.ndarray | None = None,
    measured_velocity: float | np.ndarray | None = None,
    measured_linear_velocity: Any | None = None,
    measured_angular_velocity: Any | None = None,
    projected_gravity: Any | None = None,
    skill_id: int | np.ndarray | None = None,
    command_skill_id: int | np.ndarray | None = None,
    pose_weight: float = 1.0,
    velocity_weight: float = 0.0,
    command_weight: float = 0.0,
    pose_sigma: float = 1.0,
    velocity_sigma: float = 0.25,
    commanded_yaw_rate: float | np.ndarray | None = None,
    measured_yaw_rate: float | np.ndarray | None = None,
    yaw_weight: float = 0.0,
    yaw_sigma: float = 0.25,
    lin_vel_z_weight: float = 0.0,
    ang_vel_xy_weight: float = 0.0,
    flat_orientation_weight: float = 0.0,
    base_height: float | np.ndarray | None = None,
    target_base_height: float | np.ndarray | None = None,
    height_weight: float = 0.0,
    height_sigma: float = 0.05,
) -> float | np.ndarray:
    """
    计算论文中的 task reward r_T。

    当前实现采用“按 skill 分派”的方式：
    - `walk` 技能：
      使用更贴近 PASIST 风格的手工任务奖励，
      主要由速度跟踪和直立姿态组成
    - 其它技能：
      暂时回退到通用的姿态匹配近似版本，避免在多技能尚未实现前直接失效

    参数:
    重要说明:
    - 对 `walk` 而言，`target_pose` 当前不再直接参与 `r_T` 主体计算，
      它主要用于：
      - trajectory selector 中的 DTW
      - SIL / keyframe 约束
    - 这更接近 PASIST 中“task reward 手工设计、pose 相似性用于轨迹筛选”的分工

    输入形状:
    - 单样本模式：
      - `imitation_observation`: [D]
      - `target_pose`: [D]
      - 其他量为标量
    - batched 模式：
      - `imitation_observation`: [B, D]
      - `target_pose`: [D] 或 [B, D]
      - 其他量为标量或 [B]

    返回:
    - 单样本输入时返回 `float`
    - batched 输入时返回 shape = [B] 的 `np.ndarray`

    说明:
    - 这版实现已经兼容 Isaac Lab 并行环境常见的 batched 输入
    - 如果上游只处理单环境，你仍然会得到熟悉的标量输出
    """
    if _all_skill_ids_equal(skill_id, expected_skill_id=0):
        return compute_walk_task_reward(
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

    # 对于尚未专门定义 task reward 的技能，先回退到旧的通用近似版本，
    # 这样后续扩 crawl / stilt / bipedalize 时不会立刻把整个训练链打断。
    reward = np.asarray(
        float(pose_weight)
        * pose_tracking_reward(
            imitation_observation=imitation_observation,
            target_pose=target_pose,
            sigma=pose_sigma,
        ),
        dtype=np.float32,
    )

    if commanded_velocity is not None and measured_velocity is not None and velocity_weight != 0.0:
        reward = reward + np.asarray(
            float(velocity_weight)
            * velocity_tracking_reward(
                commanded_velocity=commanded_velocity,
                measured_velocity=measured_velocity,
                sigma=velocity_sigma,
            ),
            dtype=np.float32,
        )

    if skill_id is not None and command_skill_id is not None and command_weight != 0.0:
        reward = reward + np.asarray(
            float(command_weight)
            * command_consistency_reward(
                skill_id=skill_id,
                command_skill_id=command_skill_id,
            ),
            dtype=np.float32,
        )

    return _maybe_scalar(reward)
