from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from dataclasses import fields
from datetime import datetime
from typing import Sequence
import numpy as np

import yaml


def _add_project_root_to_path() -> pathlib.Path:
    """
    确保直接运行 `python training/train.py` 时，项目根目录在 `sys.path` 中。
    """
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def _parse_skill_names(value: str) -> list[str]:
    """
    解析 skill 名称列表。

    输入:
    - `value`: 逗号分隔的 skill 名称字符串，例如 `"walk,crawl"`

    输出:
    - `list[str]`
    """
    skill_names = [item.strip() for item in value.split(",") if item.strip()]
    if not skill_names:
        raise ValueError("至少需要提供一个 skill 名称")
    return skill_names


def _print_run_summary(
    task: str,
    num_envs: int,
    iterations: int,
    enable_sil: bool,
    use_skill_selector: bool,
    skill_names: Sequence[str],
    config_path: str,
) -> None:
    """
    打印本次训练运行摘要，方便确认参数是否符合预期。
    """
    print("[INFO] Training setup")
    labels = [
        "task",
        "num_envs",
        "iterations",
        "enable_sil",
        "use_skill_selector",
        "skill_names",
        "config_path",
    ]
    label_width = max(len(label) for label in labels)
    print(f"{'task':>{label_width}}:{task}")
    print(f"{'num_envs':>{label_width}}:{num_envs}")
    print(f"{'iterations':>{label_width}}:{iterations}")
    print(f"{'enable_sil':>{label_width}}:{enable_sil}")
    print(f"{'use_skill_selector':>{label_width}}:{use_skill_selector}")
    print(f"{'skill_names':>{label_width}}:{list(skill_names)}")
    print(f"{'config_path':>{label_width}}:{config_path}")



def _load_config_path(config_path: str) -> dict:
    """
    读取“配置路径索引文件”。

    这个文件的作用不是直接存训练超参数，而是告诉训练入口：
    - PPO 配置文件在哪里
    - discriminator 配置文件在哪里
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config_path_dict = yaml.safe_load(file) or {}
    if not isinstance(config_path_dict, dict):
        raise TypeError(f"配置路径文件必须解析为字典，实际得到: {type(config_path_dict)}")
    return config_path_dict


def _parse_hidden_dims_from_config(value) -> tuple[int, ...]:
    """
    将 YAML 中的隐藏层配置统一整理成 `tuple[int, ...]`。

    支持:
    - `[128, 128]`
    - `(128, 128, 128)` 这种字符串
    - `"256,256"` 这种字符串
    """
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    if isinstance(value, str):
        stripped = value.strip().strip("()[]")
        if not stripped:
            return ()
        return tuple(int(item.strip()) for item in stripped.split(",") if item.strip())
    raise TypeError(f"无法解析隐藏层配置: {value!r}")


def _coerce_bool_string(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"无法解析布尔值字符串: {value!r}")


def _coerce_value_like(reference_value, candidate_value):
    """
    按参考值的类型，把配置里的字符串纠正成真正的数值/布尔类型。

    主要用于修复 YAML 对某些科学计数法（例如 `5e-3`）解析成字符串的情况。
    """
    if candidate_value is None:
        return None
    if isinstance(candidate_value, str):
        if isinstance(reference_value, bool):
            return _coerce_bool_string(candidate_value)
        if isinstance(reference_value, int) and not isinstance(reference_value, bool):
            return int(candidate_value.strip())
        if isinstance(reference_value, float):
            return float(candidate_value.strip())
    return candidate_value


def _coerce_trainer_config_types(config_dict: dict, trainer_config_type) -> dict:
    """
    根据 `PPOTrainerConfig` 的默认字段类型，纠正 YAML 解析结果。
    """
    defaults = trainer_config_type()
    normalized = dict(config_dict)
    for field in fields(trainer_config_type):
        field_name = field.name
        if field_name not in normalized:
            continue
        normalized[field_name] = _coerce_value_like(
            getattr(defaults, field_name),
            normalized[field_name],
        )
    return normalized


def _load_ppo_config(config_path: str) -> tuple[object, set[str]]:
    """
    从 YAML 文件加载 `PPOTrainerConfig`。
    """
    from rl.ppo_trainer import PPOTrainerConfig

    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    ppo_config = dict(config_dict["ppo"])
    ppo_config = _coerce_trainer_config_types(ppo_config, PPOTrainerConfig)

    # YAML 中的隐藏层经常写成 "(128, 128, 128)" 这种字符串。
    # 这里统一转成 tuple[int, ...]，避免后面构建 MLP 时把字符串按字符处理。
    ppo_config["actor_hidden_dims"] = _parse_hidden_dims_from_config(
        ppo_config.get("actor_hidden_dims")
    )
    ppo_config["critic_hidden_dims"] = _parse_hidden_dims_from_config(
        ppo_config.get("critic_hidden_dims")
    )

    return PPOTrainerConfig(**ppo_config), set(ppo_config.keys())


def _load_seed_from_ppo_config(config_path: str) -> int | None:
    """
    从 `ppo.yaml` 顶层可选的 `seed.seed` 读取默认随机种子。

    约定：
    - 如果命令行显式传了 `--seed`，则命令行为准
    - 否则使用这里的配置值作为默认 seed
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    if not isinstance(config_dict, dict):
        return None

    seed_section = config_dict.get("seed")
    if not isinstance(seed_section, dict):
        return None

    seed_value = seed_section.get("seed")
    if seed_value is None:
        return None
    return int(seed_value)


def _load_reward_weight_config(config_path: str | None) -> dict:
    """
    从 YAML 文件加载 reward 权重配置。

    支持两种形式：
    - 顶层直接就是权重字典
    - 顶层包一层 `reward_weights:`
    """
    if not config_path:
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    if not isinstance(config_dict, dict):
        raise TypeError(f"reward 权重配置必须解析为字典，实际得到: {type(config_dict)}")
    return config_dict


def _load_skill_one_hot_config(config_path: str | None) -> dict[int, np.ndarray]:
    """
    从 YAML 文件加载 skill_id -> one-hot 编码映射。

    支持格式：
    ```yaml
    id:
      "0": [1, 0, 0, 0]
      "1": [0, 1, 0, 0]
    ```
    """
    if not config_path:
        return {}

    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    if not isinstance(config_dict, dict):
        raise TypeError(f"skill one-hot 配置必须解析为字典，实际得到: {type(config_dict)}")

    mapping = config_dict.get("id", config_dict)
    if not isinstance(mapping, dict):
        raise TypeError("skill one-hot 配置中的 `id` 字段必须是字典")

    normalized: dict[int, np.ndarray] = {}
    for skill_id, one_hot in mapping.items():
        vector = np.asarray(one_hot, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError(f"skill_id={skill_id} 的 one-hot 不能为空")
        normalized[int(skill_id)] = vector
    return normalized


def _flatten_reward_section(
    section_config: dict | None,
    skill_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """
    将单个 reward 配置段整理成可直接覆盖 trainer config 的扁平字典。

    支持三种写法：
    1. 直接写字段
    2. 写 `common`
    3. 按 skill 名称分段，例如 `task.walk`
    """
    if not isinstance(section_config, dict):
        return {}

    flattened: dict[str, object] = {}

    for key, value in section_config.items():
        if not isinstance(value, dict):
            flattened[key] = value

    common_config = section_config.get("common")
    if isinstance(common_config, dict):
        flattened.update(common_config)

    if skill_names is not None and len(skill_names) == 1:
        skill_config = section_config.get(skill_names[0])
        if isinstance(skill_config, dict):
            flattened.update(skill_config)

    return flattened


def _apply_reward_weight_config(
    trainer_config,
    reward_weight_config: dict,
    skill_names: Sequence[str] | None = None,
    protected_keys: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    将 reward YAML 中的字段覆盖到 `PPOTrainerConfig`。

    约定：
    - `ppo.yaml` 是训练超参数主配置
    - `weight.yaml` 只补充或覆盖 `ppo.yaml` 未显式声明的 reward 相关字段
    """
    if not reward_weight_config:
        return [], []

    root = reward_weight_config.get("reward_weights", reward_weight_config)
    if not isinstance(root, dict):
        raise TypeError("reward_weights 必须是字典")

    flattened_overrides: dict[str, object] = {}

    task_config = root.get("task")
    regularization_config = root.get("regularization")
    total_config = root.get("total")

    if task_config is not None:
        flattened_overrides.update(
            _flatten_reward_section(task_config, skill_names=skill_names)
        )
    if regularization_config is not None:
        flattened_overrides.update(
            _flatten_reward_section(regularization_config, skill_names=skill_names)
        )
    if total_config is not None:
        flattened_overrides.update(
            _flatten_reward_section(total_config, skill_names=skill_names)
        )

    for key, value in root.items():
        if key in {"task", "regularization", "total"}:
            continue
        if not isinstance(value, dict):
            flattened_overrides[key] = value

    protected_keys = protected_keys or set()
    applied_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, value in flattened_overrides.items():
        if not hasattr(trainer_config, key):
            continue
        if key in protected_keys:
            skipped_keys.append(key)
            continue
        coerced_value = _coerce_value_like(getattr(trainer_config, key), value)
        setattr(trainer_config, key, coerced_value)
        applied_keys.append(key)
    return applied_keys, skipped_keys


def _to_serializable(value):
    """
    将训练统计中的 numpy / 标量对象整理成可写入 JSON 的基础类型。
    """
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _create_run_dir(log_root: pathlib.Path, task: str) -> pathlib.Path:
    """
    在指定日志根目录下创建本次训练的日志目录。

    目录结构:
    - <log_root>/<task>/<timestamp>/
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_task = task.replace("/", "_")
    run_dir = log_root / safe_task / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_run_metadata(
    run_dir: pathlib.Path,
    args: argparse.Namespace,
    trainer_config,
    config_path_dict: dict,
    discriminator_config: dict | None = None,
) -> None:
    """
    把本次训练的核心元信息写入日志目录，便于后续复现实验。
    """
    metadata = {
        "args": {key: _to_serializable(value) for key, value in vars(args).items()},
        "trainer_config": {
            key: _to_serializable(value) for key, value in vars(trainer_config).items()
        },
        "config_path_dict": _to_serializable(config_path_dict),
    }
    if discriminator_config is not None:
        metadata["discriminator_config"] = _to_serializable(discriminator_config)
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def _build_iteration_logger(run_dir: pathlib.Path):
    """
    创建一个每个 iteration 调用一次的日志函数。

    输出内容:
    - `metrics.jsonl`: 逐行保存每轮统计
    - TensorBoard event 文件: 若环境已安装 tensorboard，则同步写入
    """
    metrics_path = run_dir / "metrics.jsonl"

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    except Exception:
        writer = None

    def _log_iteration(stats: dict[str, float]) -> None:
        serializable_stats = {key: _to_serializable(value) for key, value in stats.items()}
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(serializable_stats, ensure_ascii=False) + "\n")

        if writer is not None:
            step = int(float(serializable_stats.get("iteration", 0.0)))
            for key, value in serializable_stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, float(value), step)
            writer.flush()

    return _log_iteration, writer


def _format_seconds(seconds: float) -> str:
    """
    将秒数格式化成紧凑可读形式。
    """
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes):02d}m {sec:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours):02d}h {int(minutes):02d}m {sec:04.1f}s"


def _module_for_display(module):
    """
    返回更适合打印到终端的网络结构视图。
    """
    if hasattr(module, "as_sequential"):
        try:
            return module.as_sequential()
        except Exception:
            return module
    return module


def _append_aligned_block(lines: list[str], label: str, value, label_width: int = 24) -> None:
    """
    将 `label : value` 形式的文本追加到输出列表中，并对多行值做缩进对齐。
    """
    value_lines = str(value).splitlines() or [""]
    lines.append(f"{label:>{label_width}}:{value_lines[0]}")
    continuation_prefix = " " * (label_width + 1)
    for extra_line in value_lines[1:]:
        lines.append(f"{continuation_prefix}{extra_line}")


def _print_network_overview(trainer, seed: int) -> None:
    """
    打印更接近 IsaacLab 官方风格的网络结构摘要。
    """
    lines: list[str] = ["[INFO]: Completed setting up the environment..."]
    _append_aligned_block(lines, "Actor MLP", _module_for_display(trainer.policy.mean_net))
    _append_aligned_block(lines, "Critic MLP", _module_for_display(trainer.value_function.value_net))
    if getattr(trainer, "discriminator", None) is not None:
        _append_aligned_block(lines, "Discriminator MLP", _module_for_display(trainer.discriminator.backbone))
    _append_aligned_block(lines, "Setting seed", seed)
    print("\n".join(lines))


def _print_iteration_summary(
    stats: dict[str, float],
    total_iterations: int,
    run_start_time: float,
) -> None:
    """
    以 IsaacLab 风格在控制台打印 iteration 摘要。
    """
    width = 96
    summary_labels = [
        "Computation",
        "Value function loss",
        "Surrogate loss",
        "Mean action noise std",
        "Mean reward",
        "Mean episode length",
        "Episode Reward/task",
        "Episode Reward/reg",
        "Episode Reward/sil",
        "Rollout Reward/task",
        "Rollout Reward/reg",
        "Rollout Reward/sil",
        "Rollout omega_t",
        "Rollout omega_sil",
        "SIL buffer trajectories",
        "SIL buffer mean DTW",
        "Trajectory accepted",
        "Trajectory evaluated",
        "Discriminator loss",
        "Discriminator expert score",
        "Discriminator policy score",
        "Total timesteps",
        "Iteration time",
        "Total time",
        "ETA",
    ]
    label_width = max(len(label) for label in summary_labels)
    iteration_index = int(float(stats.get("iteration", 0.0))) + 1
    collection_time = float(stats.get("iteration_collection_time_sec", 0.0))
    learning_time = float(stats.get("iteration_learning_time_sec", 0.0))
    iteration_time = float(stats.get("iteration_total_time_sec", collection_time + learning_time))
    steps_per_sec = float(stats.get("iteration_steps_per_sec", 0.0))
    total_env_steps = int(float(stats.get("total_env_steps", 0.0)))
    elapsed_time = time.perf_counter() - run_start_time
    remaining_iterations = max(total_iterations - iteration_index, 0)
    average_iteration_time = elapsed_time / max(iteration_index, 1)
    eta_seconds = average_iteration_time * remaining_iterations

    mean_reward = float(stats.get("episode_reward_mean", stats.get("rollout_reward_total_mean", 0.0)))
    mean_episode_length = float(stats.get("episode_length_mean", 0.0))

    lines: list[str] = []
    lines.append("#" * width)
    lines.append(f"Learning iteration {iteration_index}/{total_iterations}")
    lines.append("#" * width)
    _append_aligned_block(
        lines,
        "Computation",
        f"{steps_per_sec:.0f} steps/s (collection: {collection_time:.3f}s, learning: {learning_time:.3f}s)",
        label_width=label_width,
    )
    _append_aligned_block(
        lines,
        "Value function loss",
        f"{float(stats.get('ppo_value_loss', 0.0)):.4f}",
        label_width=label_width,
    )
    _append_aligned_block(
        lines,
        "Surrogate loss",
        f"{float(stats.get('ppo_policy_loss', 0.0)):.4f}",
        label_width=label_width,
    )
    _append_aligned_block(
        lines,
        "Mean action noise std",
        f"{float(stats.get('mean_action_noise_std', 0.0)):.2f}",
        label_width=label_width,
    )
    _append_aligned_block(lines, "Mean reward", f"{mean_reward:.4f}", label_width=label_width)
    _append_aligned_block(lines, "Mean episode length", f"{mean_episode_length:.2f}", label_width=label_width)

    optional_metric_specs = [
        ("Episode Reward/task", "episode_reward_task_mean", 4),
        ("Episode Reward/reg", "episode_reward_reg_mean", 4),
        ("Episode Reward/sil", "episode_reward_sil_mean", 4),
        ("Rollout Reward/task", "rollout_reward_task_mean", 4),
        ("Rollout Reward/reg", "rollout_reward_reg_mean", 4),
        ("Rollout Reward/sil", "rollout_reward_sil_mean", 4),
        ("Rollout omega_t", "rollout_omega_t_mean", 4),
        ("Rollout omega_sil", "rollout_omega_sil_mean", 4),
        ("SIL buffer trajectories", "sil_buffer_num_trajectories", 0),
        ("SIL buffer mean DTW", "sil_buffer_mean_dtw", 4),
        ("Trajectory accepted", "trajectory_accepted_count", 0),
        ("Trajectory evaluated", "trajectory_evaluated_count", 0),
        ("Discriminator ready", "discriminator_buffer_ready", 0),
        ("Discriminator batch", "discriminator_effective_batch_size", 0),
        ("Discriminator expert pool", "discriminator_expert_transition_pool", 0),
        ("Discriminator loss", "discriminator_loss", 4),
        ("Discriminator expert score", "discriminator_expert_score", 4),
        ("Discriminator policy score", "discriminator_policy_score", 4),
    ]
    for label, key, precision in optional_metric_specs:
        if key not in stats:
            continue
        value = float(stats[key])
        formatted_value = f"{value:.{precision}f}" if precision > 0 else f"{int(round(value))}"
        _append_aligned_block(lines, label, formatted_value, label_width=label_width)

    lines.append("-" * width)
    _append_aligned_block(lines, "Total timesteps", total_env_steps, label_width=label_width)
    _append_aligned_block(lines, "Iteration time", _format_seconds(iteration_time), label_width=label_width)
    _append_aligned_block(lines, "Total time", _format_seconds(elapsed_time), label_width=label_width)
    _append_aligned_block(lines, "ETA", _format_seconds(eta_seconds), label_width=label_width)

    print("\n".join(lines))

def _load_keyframe_pose_bank(path) -> dict[int, np.ndarray]:
    """
    加载 target pose bank。

    支持：
    - 一维数组：读取最后一位作为 `skill_id`
    - 二维数组：每一行读取最后一位作为 `skill_id`

    当前 target pose 约定：
    - `[base_height, joint_pos_rel(12), skill_id]`
    """
    pose = np.load(path).astype(np.float32)
    if pose.ndim == 1:
        row = pose.reshape(-1)
        return {int(round(float(row[-1]))): row}
    if pose.ndim == 2:
        pose_bank: dict[int, np.ndarray] = {}
        for index in range(pose.shape[0]):
            row = pose[index].reshape(-1)
            skill_id = int(round(float(row[-1])))
            if skill_id in pose_bank:
                raise ValueError(f"target_pose_bank 中出现重复 skill_id={skill_id}")
            pose_bank[skill_id] = row
        return pose_bank
    raise ValueError(f"target_pose_bank 只支持 1D 或 2D 数组，实际得到 shape={pose.shape}")

def main() -> None:
    parser = argparse.ArgumentParser(description="运行当前项目的 PASIST PPO 训练入口。")

    # 环境与运行时参数：这些参数保留在命令行最合适。
    parser.add_argument("--task", type=str, default="PASIST-Go2-Base", help="要训练的 Gym task id。")
    parser.add_argument("--num-envs", type=int, default=1, help="并行环境数量。建议先从 1 开始。")
    parser.add_argument("--skill-names", type=str, default="walk", help="逗号分隔的 skill 名称列表。")
    parser.add_argument("--seed", type=int, default=0, help="首次 reset 使用的随机种子。")
    parser.add_argument("--iterations", type=int, default=100, help="训练 iteration 数量。")

    # 训练配置来源：PPO / reward / discriminator 等超参数由 YAML 管理。
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/config_path.yaml",
        help="训练配置路径索引 YAML 文件路径。",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="训练日志根目录。默认写入项目根目录下的 logs 文件夹。",
    )

    # 这些布尔开关保留为命令行覆盖项，方便快速实验。
    parser.add_argument("--disable-sil", dest="enable_sil", action="store_false", help="关闭 SIL。")
    parser.add_argument(
        "--disable-skill-selector",
        dest="use_skill_selector",
        action="store_false",
        help="关闭 SkillSelector。",
    )
    parser.set_defaults(enable_sil=True, use_skill_selector=True)
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default="checkpoints",
        help="保存模型的目录。相对路径时会保存在本次 run_dir 下。",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="每隔多少个 iteration 保存一次模型。设为 0 表示只保存最终模型。",
    )
    parser.add_argument(
        "--target-pose-bank",
        type=str,
        default="target_pose_bank/target_pose_bank.npy",
        help="target pose bank 路径；默认读取项目内生成的 target_pose_bank.npy。",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    project_root = _add_project_root_to_path()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    tb_writer = None

    try:
        # Isaac Sim App 启动后再导入环境与训练模块，避免初始化顺序问题。
        import envs.isaacsim_mini_envs  # noqa: F401
        import envs.pasist_env_cfg  # noqa: F401
        import torch
        from envs.IsaacLabPasistEnv import IsaacLabPasistEnv
        from rl.ppo_trainer import PPOTrainer
        from sil.discriminator import load_discriminator_config
        from sil.skill_selector import SkillSelector

        skill_names = _parse_skill_names(args.skill_names)

        def save_model(trainer: object, save_model_dir: str, name: str):
            """
            保存模型到指定目录。

            当前保存内容包括：
            - policy / value 网络参数
            - actor / critic optimizer 状态
            - 可选的 discriminator 与其 optimizer 状态
            - 当前 trainer 配置

            注意：
            - 这里不直接保存 `history`，因为按 iteration 中途保存时，
              外层 `history` 还没有最终生成完。
            """
            save_path = pathlib.Path(save_model_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "policy_state_dict": trainer.policy.state_dict(),
                # "value_state_dict": trainer.value_function.state_dict(),
                # "actor_optimizer_state_dict": trainer.actor_optimizer.state_dict(),
                # "critic_optimizer_state_dict": trainer.critic_optimizer.state_dict(),
                # "config": vars(ppotrainer_config),
            }

            if getattr(trainer, "discriminator", None) is not None:
                checkpoint["discriminator_state_dict"] = trainer.discriminator.state_dict()
            if getattr(trainer, "discriminator_optimizer", None) is not None:
                checkpoint["discriminator_optimizer_state_dict"] = (
                    trainer.discriminator_optimizer.state_dict()
                )

            torch.save(
                checkpoint,
                save_path / f"{name}.pt",
            )
            print(f"[INFO] Model saved to: {save_path / f'{name}.pt'}")


        # load all configs paths
        config_path_dict = _load_config_path(args.config_path)
        ppo_config_path = config_path_dict["ppo_config_path"]["PPOTrainer_config_path"]
        discriminator_config_path = config_path_dict["discriminator_config_path"]["Discriminator_config_path"]
        reward_weight_config_path = (
            config_path_dict.get("reward_weight_config_path", {}).get("RewardWeight_config_path")
        )
        skill_one_hot_config_path = (
            config_path_dict.get("skill_one_hot_config_path", {}).get("SkillOneHot_config_path")
        )

        seed_from_cli = any(
            token == "--seed" or token.startswith("--seed=")
            for token in sys.argv[1:]
        )
        if not seed_from_cli:
            config_seed = _load_seed_from_ppo_config(ppo_config_path)
            if config_seed is not None:
                args.seed = int(config_seed)

        ppotrainer_config, ppo_defined_keys = _load_ppo_config(ppo_config_path)
        applied_reward_weight_keys, skipped_reward_weight_keys = _apply_reward_weight_config(
            ppotrainer_config,
            _load_reward_weight_config(reward_weight_config_path),
            skill_names=skill_names,
            protected_keys=ppo_defined_keys,
        )
        ppotrainer_config.device = args.device
        discriminator_config_source = "config_path.yaml"
        if not getattr(ppotrainer_config, "discriminator_config_path", None):
            ppotrainer_config.discriminator_config_path = discriminator_config_path
        else:
            discriminator_config_source = "ppo.yaml"
        ppotrainer_config.enable_sil = bool(args.enable_sil and ppotrainer_config.enable_sil)
        ppotrainer_config.use_skill_selector = bool(args.use_skill_selector and ppotrainer_config.use_skill_selector)
        _print_run_summary(
            task=args.task,
            num_envs=args.num_envs,
            iterations=args.iterations,
            enable_sil=bool(ppotrainer_config.enable_sil),
            use_skill_selector=bool(ppotrainer_config.use_skill_selector),
            skill_names=skill_names,
            config_path=args.config_path,
        )
        if reward_weight_config_path:
            print(f"[INFO] Reward weight config: {reward_weight_config_path}")
            if applied_reward_weight_keys:
                print(f"[INFO] Applied reward weight keys: {', '.join(sorted(applied_reward_weight_keys))}")
            if skipped_reward_weight_keys:
                print(
                    "[INFO] Reward weight keys skipped because ppo.yaml has priority: "
                    + ", ".join(sorted(skipped_reward_weight_keys))
                )
            if not applied_reward_weight_keys and not skipped_reward_weight_keys:
                print("[INFO] Reward weight config loaded, but no matching trainer fields were overridden")
        if skill_one_hot_config_path:
            print(f"[INFO] Skill one-hot config: {skill_one_hot_config_path}")
        if not seed_from_cli:
            print(f"[INFO] Seed resolved from ppo.yaml/default: {args.seed}")
        else:
            print(f"[INFO] Seed resolved from CLI: {args.seed}")
        print(f"[INFO] Trainer device resolved from AppLauncher/CLI: {ppotrainer_config.device}")
        print(
            f"[INFO] Discriminator config path resolved from {discriminator_config_source}: "
            f"{ppotrainer_config.discriminator_config_path}"
        )


        log_root = pathlib.Path(args.log_dir)
        if not log_root.is_absolute():
            log_root = project_root / log_root
        run_dir = _create_run_dir(log_root=log_root, task=args.task)
        iteration_logger, tb_writer = _build_iteration_logger(run_dir)
        discriminator_config_for_metadata = load_discriminator_config(
            ppotrainer_config.discriminator_config_path
        )
        _write_run_metadata(
            run_dir,
            args=args,
            trainer_config=ppotrainer_config,
            config_path_dict=config_path_dict,
            discriminator_config=discriminator_config_for_metadata,
        )
        print(f"[INFO] Log directory: {run_dir}")
        if tb_writer is not None:
            print(f"[INFO] TensorBoard directory: {run_dir / 'tensorboard'}")
        else:
            print("[WARNING] TensorBoard writer unavailable; only metrics.jsonl will be saved")

        save_model_dir = pathlib.Path(args.save_model_dir)
        if not save_model_dir.is_absolute():
            save_model_dir = run_dir / save_model_dir
        print(f"[INFO] Checkpoint directory: {save_model_dir}")

        env = IsaacLabPasistEnv(
            task_id=args.task,
            num_envs=args.num_envs,
            skill_names=skill_names,
            target_pose_bank=_load_keyframe_pose_bank(args.target_pose_bank) if args.target_pose_bank else None,
            skill_one_hot_map=_load_skill_one_hot_config(skill_one_hot_config_path),
        )

        skill_selector = None
        if ppotrainer_config.use_skill_selector:
            skill_selector = SkillSelector(
                num_skills=env.num_skills,
                velocity_range=env.velocity_range,
            )
        
        

        trainer = PPOTrainer(
            env=env,
            config=ppotrainer_config,
            skill_selector=skill_selector,
        )
        _print_network_overview(trainer=trainer, seed=args.seed)
        run_start_time = time.perf_counter()

        def on_iteration_end(stats: dict[str, float]) -> None:
            """
            组合 iteration 日志与定期 checkpoint 保存逻辑。

            调用顺序：
            1. 先记录 metrics.jsonl / TensorBoard
            2. 再根据 iteration 编号判断是否保存模型
            """
            iteration_logger(stats)
            _print_iteration_summary(
                stats=stats,
                total_iterations=args.iterations,
                run_start_time=run_start_time,
            )

            if args.save_interval <= 0:
                return

            iteration_index = int(float(stats.get("iteration", -1)))
            if iteration_index < 0:
                return

            # iteration 从 0 开始计数，更符合用户直觉的是第 1、10、20... 轮。
            iterations_nums = iteration_index + 1
            if iterations_nums % args.save_interval == 0:
                save_model(
                    trainer=trainer,
                    save_model_dir=str(save_model_dir),
                    name=f"iter_{iterations_nums:06d}",
                )

        history = trainer.train(
            num_iterations=args.iterations,
            seed=args.seed,
            on_iteration_end=on_iteration_end,
        )

        save_model(
            trainer=trainer,
            save_model_dir=str(save_model_dir),
            name="final",
        )

        with (run_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump([_to_serializable(item) for item in history], file, ensure_ascii=False, indent=2)

        print("[OK] Training finished")
        print(f"[INFO] Training artifacts saved to: {run_dir}")

    except Exception as exc:
        print("[ERROR] Training pipeline failed")
        print(f"type: {type(exc).__name__}")
        print(f"message: {exc}")
        traceback.print_exc()
        raise

    finally:
        if env is not None:
            env.close()
        if tb_writer is not None:
            tb_writer.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
