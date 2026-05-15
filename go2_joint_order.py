from __future__ import annotations

from collections.abc import Sequence


# 当前 PASIST / target_pose_bank / deploy 侧约定的 Go2 12 关节顺序。
# 这个顺序来自工程现有的 policy pose 表达，而不是 Unitree URDF 原始 dof 顺序。
GO2_JOINT_ORDER: tuple[str, ...] = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

# 当前 `unitree_rl_gym` Go2 asset 的实际 dof 顺序。
# 这里保留为显式常量，便于测试和排查；运行时 backend 仍会从 `env.dof_names` 再校验一次。
UNITREE_GO2_DOF_ORDER: tuple[str, ...] = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


def build_reorder_indices(
    source_order: Sequence[str],
    target_order: Sequence[str],
) -> tuple[int, ...]:
    """
    返回把 `source_order` 重排成 `target_order` 所需的下标序列。

    用法：
    - `reordered = value[..., build_reorder_indices(source, target)]`
    """
    source = tuple(str(name) for name in source_order)
    target = tuple(str(name) for name in target_order)

    if len(source) != len(target):
        raise ValueError(
            f"joint order 长度不一致：source={len(source)}，target={len(target)}"
        )
    if len(set(source)) != len(source):
        raise ValueError(f"source_order 存在重复关节名: {source}")
    if len(set(target)) != len(target):
        raise ValueError(f"target_order 存在重复关节名: {target}")

    source_set = set(source)
    target_set = set(target)
    if source_set != target_set:
        missing_in_source = sorted(target_set - source_set)
        missing_in_target = sorted(source_set - target_set)
        raise ValueError(
            "joint order 名称集合不一致："
            f" missing_in_source={missing_in_source},"
            f" missing_in_target={missing_in_target}"
        )

    source_index = {name: index for index, name in enumerate(source)}
    return tuple(source_index[name] for name in target)
