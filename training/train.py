from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback
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
    print(f"task: {task}")
    print(f"num_envs: {num_envs}")
    print(f"iterations: {iterations}")
    print(f"enable_sil: {enable_sil}")
    print(f"use_skill_selector: {use_skill_selector}")
    print(f"skill_names: {list(skill_names)}")
    print(f"config_path: {config_path}")



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


def _load_ppo_config(config_path: str) -> object:
    """
    从 YAML 文件加载 `PPOTrainerConfig`。
    """
    from rl.ppo_trainer import PPOTrainerConfig

    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file) or {}

    ppo_config = dict(config_dict["ppo"])

    # YAML 中的隐藏层经常写成 "(128, 128, 128)" 这种字符串。
    # 这里统一转成 tuple[int, ...]，避免后面构建 MLP 时把字符串按字符处理。
    ppo_config["actor_hidden_dims"] = _parse_hidden_dims_from_config(
        ppo_config.get("actor_hidden_dims")
    )
    ppo_config["critic_hidden_dims"] = _parse_hidden_dims_from_config(
        ppo_config.get("critic_hidden_dims")
    )

    return PPOTrainerConfig(**ppo_config)


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

def _load_keyframe_pose_bank(path) -> dict[int, np.ndarray]:
    pose = np.load(path).astype(np.float32).reshape(-1)
    return {0: pose}

def main() -> None:
    parser = argparse.ArgumentParser(description="运行当前项目的 PASIST PPO 训练入口。")

    # 环境与运行时参数：这些参数保留在命令行最合适。
    parser.add_argument("--task", type=str, default="PASIST-Go2-Base", help="要训练的 Gym task id。")
    parser.add_argument("--num-envs", type=int, default=1, help="并行环境数量。建议先从 1 开始。")
    parser.add_argument("--skill-names", type=str, default="walk", help="逗号分隔的 skill 名称列表。")
    parser.add_argument("--seed", type=int, default=0, help="首次 reset 使用的随机种子。")
    parser.add_argument("--iterations", type=int, default=100, help="训练 iteration 数量。")
    parser.add_argument(
        "--trainer-device",
        type=str,
        default=None,
        help="PPO 使用的 torch device。默认跟随 AppLauncher 提供的 --device。",
    )

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
    # 单帧路径，测试用
    parser.add_argument(
    "--target-pose-bank",
    type=str,
    default="target_pose_bank/target_pose_bank.npy",
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
        from sil.skill_selector import SkillSelector

        skill_names = _parse_skill_names(args.skill_names)
        _print_run_summary(
            task=args.task,
            num_envs=args.num_envs,
            iterations=args.iterations,
            enable_sil=bool(args.enable_sil),
            use_skill_selector=bool(args.use_skill_selector),
            skill_names=skill_names,
            config_path=args.config_path,
        )

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


        trainer_device = args.trainer_device or args.device
        # load all configs paths
        config_path_dict = _load_config_path(args.config_path)
        ppo_config_path = config_path_dict["ppo_config_path"]["PPOTrainer_config_path"]

        discriminator_config_path = config_path_dict["discriminator_config_path"]["Discriminator_config_path"]

        ppotrainer_config = _load_ppo_config(ppo_config_path)
        ppotrainer_config.device = trainer_device
        ppotrainer_config.discriminator_config_path = discriminator_config_path
        ppotrainer_config.enable_sil = bool(args.enable_sil and ppotrainer_config.enable_sil)
        ppotrainer_config.use_skill_selector = bool(args.use_skill_selector and ppotrainer_config.use_skill_selector)


        log_root = pathlib.Path(args.log_dir)
        if not log_root.is_absolute():
            log_root = project_root / log_root
        run_dir = _create_run_dir(log_root=log_root, task=args.task)
        iteration_logger, tb_writer = _build_iteration_logger(run_dir)
        _write_run_metadata(run_dir, args=args, trainer_config=ppotrainer_config, config_path_dict=config_path_dict)
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

        def on_iteration_end(stats: dict[str, float]) -> None:
            """
            组合 iteration 日志与定期 checkpoint 保存逻辑。

            调用顺序：
            1. 先记录 metrics.jsonl / TensorBoard
            2. 再根据 iteration 编号判断是否保存模型
            """
            iteration_logger(stats)

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
        if history:
            last = history[-1]
            for key in sorted(last.keys()):
                print(f"{key}: {last[key]}")

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
