from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INIT_OUTPUT_NPY = PROJECT_ROOT / "init_pose" / "init_pose.npy"
DEFAULT_INIT_OUTPUT_NPZ = PROJECT_ROOT / "init_pose" / "generated_init_pose.npz"
DEFAULT_TARGET_OUTPUT_NPY = PROJECT_ROOT / "target_pose_bank" / "target_pose_bank.npy"
DEFAULT_TARGET_OUTPUT_NPZ = PROJECT_ROOT / "target_pose_bank" / "generated_target_pose.npz"

# 顺序与工程中的 GO2_JOINT_ORDER 一致：FR -> FL -> RR -> RL，每条腿依次 hip / thigh / calf。
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

# IsaacLab / 本工程默认的 Go2 nominal joint pose。
DEFAULT_JOINT_POS = np.array(
    [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5],
    dtype=np.float32,
)


@dataclass(frozen=True)
class PoseSpec:
    """
    一组满足左右镜像、前后可不同的 12 关节姿态参数。

    说明：
    1. 左右对称：右腿 hip 为负，左腿 hip 为正。
    2. 前后允许不同：这样更接近正常四足站姿，也不意味着 pitch 一定不为 0。
    3. target_base_height_m / target_pitch_deg 是设计目标，不是解析解输入。
       当前脚本给出“适合训练的合理默认姿态”，而不是做精确 IK 反解。
    """

    name: str
    description: str
    skill_id: int
    target_base_height_m: float
    target_pitch_deg: float

    front_hip_abduction_abs: float
    front_thigh: float
    front_calf: float

    rear_hip_abduction_abs: float
    rear_thigh: float
    rear_calf: float

    @property
    def is_left_right_symmetric(self) -> bool:
        return True

    @property
    def is_front_rear_same(self) -> bool:
        return (
            np.isclose(self.front_hip_abduction_abs, self.rear_hip_abduction_abs)
            and np.isclose(self.front_thigh, self.rear_thigh)
            and np.isclose(self.front_calf, self.rear_calf)
        )


def default_init_pose_spec() -> PoseSpec:
    """
    合理默认站姿。

    设计原则：
    - 直接采用 Go2 常见 nominal standing pose
    - 左右镜像
    - 前后略有差异，以补偿前后腿安装位置差异
    - 通常能对应接近 25 cm、pitch 接近 0 的稳定站姿
    """
    return PoseSpec(
        name="init_pose",
        description="Hand-designed nominal standing pose for reset / initialization.",
        skill_id=0,
        target_base_height_m=0.25,
        target_pitch_deg=0.0,
        front_hip_abduction_abs=0.10,
        front_thigh=0.80,
        front_calf=-1.50,
        rear_hip_abduction_abs=0.10,
        rear_thigh=1.00,
        rear_calf=-1.50,
    )


def default_walk_target_pose_spec() -> PoseSpec:
    """
    合理默认 walk 目标姿态。

    设计原则：
    - 不使用你已有的录制帧
    - 仍保持左右镜像
    - 前后腿不同，但不过分激进，尽量靠近 25 cm / 0 deg 的“中性 walk core posture”
    - 比 init_pose 略微收髋、减少前后差，让它更像一个技能 keyframe，而不是纯 reset 姿态
    """
    return PoseSpec(
        name="target_pose",
        description="Hand-designed neutral walk keyframe candidate.",
        skill_id=0,
        target_base_height_m=0.25,
        target_pitch_deg=0.0,
        front_hip_abduction_abs=0.08,
        front_thigh=0.78,
        front_calf=-1.48,
        rear_hip_abduction_abs=0.08,
        rear_thigh=0.92,
        rear_calf=-1.48,
    )


def build_joint_pos(spec: PoseSpec) -> np.ndarray:
    """
    生成工程顺序下的 12 维绝对关节角。

    输出顺序：
    FR, FL, RR, RL，每条腿为 hip / thigh / calf。
    """
    joint_pos = np.array(
        [
            -spec.front_hip_abduction_abs,
            spec.front_thigh,
            spec.front_calf,
            spec.front_hip_abduction_abs,
            spec.front_thigh,
            spec.front_calf,
            -spec.rear_hip_abduction_abs,
            spec.rear_thigh,
            spec.rear_calf,
            spec.rear_hip_abduction_abs,
            spec.rear_thigh,
            spec.rear_calf,
        ],
        dtype=np.float32,
    )
    if joint_pos.shape != (12,):
        raise RuntimeError(f"生成的姿态维度异常：{joint_pos.shape}")
    return joint_pos


def projected_gravity_from_pitch_deg(pitch_deg: float) -> np.ndarray:
    """
    根据目标 pitch 构造 projected gravity。

    在当前脚本里默认 roll = 0，仅用作 target pose bank 的静态元信息。
    pitch = 0 时返回 [0, 0, -1]。
    """
    pitch_rad = np.deg2rad(float(pitch_deg))
    return np.array([np.sin(pitch_rad), 0.0, -np.cos(pitch_rad)], dtype=np.float32)


def build_imitation_target(spec: PoseSpec, joint_pos: np.ndarray) -> np.ndarray:
    """
    生成与当前工程 pose-only imitation_obs_dim 对齐的 14 维 target pose。

    对齐顺序：
    - base_height       (1,)
    - joint_pos_rel     (12,)
    - skill_id          (1,)
    """
    base_height = np.asarray([spec.target_base_height_m], dtype=np.float32)
    skill_id = np.asarray([float(spec.skill_id)], dtype=np.float32)
    joint_pos_rel = joint_pos.astype(np.float32) - DEFAULT_JOINT_POS
    imitation_target = np.concatenate([base_height, joint_pos_rel, skill_id]).astype(np.float32)
    if imitation_target.shape != (14,):
        raise RuntimeError(f"生成的 imitation target 维度异常：{imitation_target.shape}")
    return imitation_target


def get_init_pose(spec: PoseSpec | None = None) -> np.ndarray:
    """供其他脚本 import 调用。"""
    return build_joint_pos(spec or default_init_pose_spec())


def get_target_frame(spec: PoseSpec | None = None) -> np.ndarray:
    """
    供其他脚本 import 调用。

    返回的是训练侧真正使用的 14 维 pose-only target pose。
    """
    pose_spec = spec or default_walk_target_pose_spec()
    joint_pos = build_joint_pos(pose_spec)
    return build_imitation_target(pose_spec, joint_pos)


def build_named_joint_dict(joint_pos: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(GO2_JOINT_ORDER, joint_pos)}


def build_metadata(spec: PoseSpec, joint_pos: np.ndarray, imitation_target: np.ndarray | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "name": spec.name,
        "description": spec.description,
        "skill_id": int(spec.skill_id),
        "joint_order": list(GO2_JOINT_ORDER),
        "target_base_height_m": float(spec.target_base_height_m),
        "target_pitch_deg": float(spec.target_pitch_deg),
        "left_right_symmetric": bool(spec.is_left_right_symmetric),
        "front_rear_same": bool(spec.is_front_rear_same),
        "default_joint_pos": DEFAULT_JOINT_POS.tolist(),
        "joint_pos": joint_pos.tolist(),
        "joint_pos_rel": (joint_pos - DEFAULT_JOINT_POS).tolist(),
        "base_height": float(spec.target_base_height_m),
        "pitch_deg": float(spec.target_pitch_deg),
        "pitch_rad": float(np.deg2rad(spec.target_pitch_deg)),
        "projected_gravity": projected_gravity_from_pitch_deg(spec.target_pitch_deg).tolist(),
        "named_joint_pos": build_named_joint_dict(joint_pos),
        "spec": asdict(spec),
        "note": (
            "该文件保存的是手工设计的合理默认姿态。"
            "init_pose 为 12 维绝对关节角；target_pose_bank 为 14 维 pose-only target，"
            "顺序为 [base_height, joint_pos_rel(12), skill_id]，与当前工程训练接口直接对齐。"
        ),
    }
    if imitation_target is not None:
        metadata["imitation_target"] = imitation_target.tolist()
    return metadata


def save_init_pose(joint_pos: np.ndarray, spec: PoseSpec, output_npy: Path, output_npz: Path) -> None:
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(spec, joint_pos)
    np.save(output_npy, joint_pos.reshape(1, -1))
    np.savez(
        output_npz,
        joint_pos=joint_pos.reshape(1, -1),
        joint_order=np.asarray(GO2_JOINT_ORDER),
        target_base_height_m=np.float32(spec.target_base_height_m),
        target_pitch_deg=np.float32(spec.target_pitch_deg),
        left_right_symmetric=np.bool_(spec.is_left_right_symmetric),
        front_rear_same=np.bool_(spec.is_front_rear_same),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, indent=2)),
    )


def save_target_pose_bank(
    imitation_target: np.ndarray,
    joint_pos: np.ndarray,
    spec: PoseSpec,
    output_npy: Path,
    output_npz: Path,
) -> None:
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(spec, joint_pos, imitation_target=imitation_target)
    np.save(output_npy, imitation_target.reshape(1, -1))
    np.savez(
        output_npz,
        imitation_target=imitation_target.reshape(1, -1),
        joint_pos=joint_pos.reshape(1, -1),
        joint_order=np.asarray(GO2_JOINT_ORDER),
        target_base_height_m=np.float32(spec.target_base_height_m),
        target_pitch_deg=np.float32(spec.target_pitch_deg),
        left_right_symmetric=np.bool_(spec.is_left_right_symmetric),
        front_rear_same=np.bool_(spec.is_front_rear_same),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, indent=2)),
    )


def add_pose_args(
    parser: argparse.ArgumentParser,
    prefix: str,
    defaults: PoseSpec,
    default_npy: Path,
    default_npz: Path,
    title: str,
) -> None:
    parser.add_argument(
        f"--{prefix}-base-height-m",
        type=float,
        default=defaults.target_base_height_m,
        help=f"{title} 目标基座高度（米）。默认 {defaults.target_base_height_m}。",
    )
    parser.add_argument(
        f"--{prefix}-pitch-deg",
        type=float,
        default=defaults.target_pitch_deg,
        help=f"{title} 目标俯仰角（度）。默认 {defaults.target_pitch_deg}。",
    )
    parser.add_argument(
        f"--{prefix}-front-hip-abduction",
        type=float,
        default=defaults.front_hip_abduction_abs,
        help=f"{title} 前腿 hip 外展绝对值。右腿取负，左腿取正。",
    )
    parser.add_argument(
        f"--{prefix}-front-thigh",
        type=float,
        default=defaults.front_thigh,
        help=f"{title} 前腿 thigh 角。",
    )
    parser.add_argument(
        f"--{prefix}-front-calf",
        type=float,
        default=defaults.front_calf,
        help=f"{title} 前腿 calf 角。",
    )
    parser.add_argument(
        f"--{prefix}-rear-hip-abduction",
        type=float,
        default=defaults.rear_hip_abduction_abs,
        help=f"{title} 后腿 hip 外展绝对值。右腿取负，左腿取正。",
    )
    parser.add_argument(
        f"--{prefix}-rear-thigh",
        type=float,
        default=defaults.rear_thigh,
        help=f"{title} 后腿 thigh 角。",
    )
    parser.add_argument(
        f"--{prefix}-rear-calf",
        type=float,
        default=defaults.rear_calf,
        help=f"{title} 后腿 calf 角。",
    )
    parser.add_argument(
        f"--{prefix}-output-npy",
        type=Path,
        default=default_npy,
        help=f"{title} 输出 .npy 路径。默认 {default_npy}",
    )
    parser.add_argument(
        f"--{prefix}-output-npz",
        type=Path,
        default=default_npz,
        help=f"{title} 输出 .npz 路径。默认 {default_npz}",
    )


def build_pose_spec_from_args(args: argparse.Namespace, prefix: str, name: str, description: str) -> PoseSpec:
    return PoseSpec(
        name=name,
        description=description,
        skill_id=0,
        target_base_height_m=float(getattr(args, f"{prefix}_base_height_m")),
        target_pitch_deg=float(getattr(args, f"{prefix}_pitch_deg")),
        front_hip_abduction_abs=float(getattr(args, f"{prefix}_front_hip_abduction")),
        front_thigh=float(getattr(args, f"{prefix}_front_thigh")),
        front_calf=float(getattr(args, f"{prefix}_front_calf")),
        rear_hip_abduction_abs=float(getattr(args, f"{prefix}_rear_hip_abduction")),
        rear_thigh=float(getattr(args, f"{prefix}_rear_thigh")),
        rear_calf=float(getattr(args, f"{prefix}_rear_calf")),
    )


def parse_args() -> argparse.Namespace:
    init_defaults = default_init_pose_spec()
    target_defaults = default_walk_target_pose_spec()

    parser = argparse.ArgumentParser(
        description=(
            "同时生成 Go2 的 init pose 和 walk target pose。"
            "init_pose 输出为 12 维绝对关节角；target_pose_bank 输出为 14 维 pose-only target。"
        )
    )
    add_pose_args(
        parser=parser,
        prefix="init",
        defaults=init_defaults,
        default_npy=DEFAULT_INIT_OUTPUT_NPY,
        default_npz=DEFAULT_INIT_OUTPUT_NPZ,
        title="init_pose",
    )
    add_pose_args(
        parser=parser,
        prefix="target",
        defaults=target_defaults,
        default_npy=DEFAULT_TARGET_OUTPUT_NPY,
        default_npz=DEFAULT_TARGET_OUTPUT_NPZ,
        title="target_pose_bank",
    )
    parser.add_argument("--print-only", action="store_true", help="只打印结果，不写文件。")
    return parser.parse_args()


def print_pose_summary(spec: PoseSpec, joint_pos: np.ndarray, imitation_target: np.ndarray | None = None) -> None:
    print(f"[INFO] {spec.name}")
    print(f"[INFO]   description           = {spec.description}")
    print(f"[INFO]   target_base_height_m  = {spec.target_base_height_m:.4f}")
    print(f"[INFO]   target_pitch_deg      = {spec.target_pitch_deg:.4f}")
    print(f"[INFO]   left_right_symmetric  = {spec.is_left_right_symmetric}")
    print(f"[INFO]   front_rear_same       = {spec.is_front_rear_same}")
    print(f"[INFO]   joint_pos             = {np.array2string(joint_pos, precision=6, separator=', ')}")
    if imitation_target is not None:
        print(f"[INFO]   imitation_target     = {np.array2string(imitation_target, precision=6, separator=', ')}")


def main() -> None:
    args = parse_args()

    init_spec = build_pose_spec_from_args(
        args,
        prefix="init",
        name="init_pose",
        description="Hand-designed nominal standing pose for reset / initialization.",
    )
    target_spec = build_pose_spec_from_args(
        args,
        prefix="target",
        name="target_pose",
        description="Hand-designed neutral walk keyframe candidate.",
    )

    init_joint_pos = build_joint_pos(init_spec)
    target_joint_pos = build_joint_pos(target_spec)
    target_imitation = build_imitation_target(target_spec, target_joint_pos)

    print("[INFO] joint_order:")
    for index, name in enumerate(GO2_JOINT_ORDER):
        print(f"  {index:02d}: {name}")
    print_pose_summary(init_spec, init_joint_pos)
    print_pose_summary(target_spec, target_joint_pos, imitation_target=target_imitation)

    if args.print_only:
        return

    save_init_pose(init_joint_pos, init_spec, args.init_output_npy, args.init_output_npz)
    save_target_pose_bank(
        target_imitation,
        target_joint_pos,
        target_spec,
        args.target_output_npy,
        args.target_output_npz,
    )

    print(f"[INFO] Saved init npy to: {args.init_output_npy}")
    print(f"[INFO] Saved init npz to: {args.init_output_npz}")
    print(f"[INFO] Saved target npy to: {args.target_output_npy}")
    print(f"[INFO] Saved target npz to: {args.target_output_npz}")


if __name__ == "__main__":
    main()
