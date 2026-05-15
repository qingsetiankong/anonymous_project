from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback


def _add_project_root_to_path() -> pathlib.Path:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def _prepare_isaacgym_runtime() -> None:
    python_bin = pathlib.Path(sys.executable).resolve().parent
    path = os.environ.get("PATH", "")
    if str(python_bin) not in path.split(os.pathsep):
        os.environ["PATH"] = str(python_bin) + os.pathsep + path
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")


def _save_model(trainer: object, save_model_dir: pathlib.Path, name: str) -> None:
    import torch

    save_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "policy_state_dict": trainer.policy.state_dict(),
        "value_state_dict": trainer.value_function.state_dict(),
        "config": vars(trainer.config),
    }
    if getattr(trainer, "discriminator", None) is not None:
        checkpoint["discriminator_state_dict"] = trainer.discriminator.state_dict()
    if getattr(trainer, "discriminator_optimizer", None) is not None:
        checkpoint["discriminator_optimizer_state_dict"] = trainer.discriminator_optimizer.state_dict()

    output_path = save_model_dir / f"{name}.pt"
    torch.save(checkpoint, output_path)
    print(f"[INFO] Model saved to: {output_path}")


def main() -> None:
    project_root = _add_project_root_to_path()
    _prepare_isaacgym_runtime()

    from numpy_compat import ensure_numpy_legacy_aliases

    ensure_numpy_legacy_aliases()

    parser = argparse.ArgumentParser(description="使用 Isaac Gym + unitree_rl_gym Go2 后端运行 PASIST PPO。")
    parser.add_argument("--task", type=str, default="go2", help="unitree_rl_gym 中注册的 task 名称。")
    parser.add_argument("--num-envs", type=int, default=4096, help="Isaac Gym 并行环境数量。")
    parser.add_argument("--skill-names", type=str, default="walk", help="逗号分隔的 skill 名称列表。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--iterations", type=int, default=100, help="训练 iteration 数量。")

    parser.add_argument("--sim-device", type=str, default="cuda:0", help="Isaac Gym 仿真设备，例如 cuda:0 或 cpu。")
    parser.add_argument("--rl-device", type=str, default="cuda:0", help="PASIST PPO 训练设备。")
    parser.add_argument("--viewer", dest="headless", action="store_false", help="打开 Isaac Gym viewer。")
    parser.set_defaults(headless=True)
    parser.add_argument("--disable-gpu-pipeline", dest="use_gpu_pipeline", action="store_false")
    parser.set_defaults(use_gpu_pipeline=True)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=10)
    parser.add_argument("--enable-noise", action="store_true", help="启用 unitree_rl_gym 原始观测噪声。")
    parser.add_argument("--enable-domain-rand", action="store_true", help="启用 unitree_rl_gym 原始 domain randomization。")
    parser.add_argument("--command-resampling-time", type=float, default=1.0e9)
    parser.add_argument(
        "--disable-episode-phase-randomization",
        dest="randomize_episode_phase",
        action="store_false",
        help="关闭全量 reset 后的 episode phase 随机化。默认开启，用于避免同步 timeout。",
    )
    parser.set_defaults(randomize_episode_phase=True)

    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/config_path.yaml",
        help="训练配置路径索引 YAML 文件。",
    )
    parser.add_argument("--log-dir", type=str, default="logs_isaacgym", help="训练日志根目录。")
    parser.add_argument("--save-model-dir", type=str, default="checkpoints", help="checkpoint 保存目录。")
    parser.add_argument("--save-interval", type=int, default=100, help="每隔多少个 iteration 保存一次模型。")
    parser.add_argument(
        "--target-pose-bank",
        type=str,
        default="target_pose_bank/target_pose_bank.npy",
        help="target pose bank 路径。传空字符串可关闭显式加载。",
    )

    parser.add_argument("--disable-sil", dest="enable_sil", action="store_false", help="关闭 SIL。")
    parser.add_argument("--disable-skill-selector", dest="use_skill_selector", action="store_false")
    parser.set_defaults(enable_sil=True, use_skill_selector=True)

    args = parser.parse_args()

    env = None
    tb_writer = None

    try:
        import isaacgym  # noqa: F401 - 必须早于 torch/rl 模块导入

        from envs.IsaacGymPasistEnv import IsaacGymPasistEnv
        from envs.isaacgym_unitree_go2_backend import make_unitree_go2_pasist_backend
        from rl.ppo_trainer import PPOTrainer
        from sil.discriminator import load_discriminator_config
        from sil.skill_selector import SkillSelector
        from training.train import (
            _apply_reward_weight_config,
            _build_iteration_logger,
            _create_run_dir,
            _load_config_path,
            _load_keyframe_pose_bank,
            _load_ppo_config,
            _load_reward_weight_config,
            _load_seed_from_ppo_config,
            _load_skill_one_hot_config,
            _parse_skill_names,
            _print_iteration_summary,
            _print_network_overview,
            _print_run_summary,
            _to_serializable,
            _write_run_metadata,
        )

        config_path_dict = _load_config_path(args.config_path)
        ppo_config_path = config_path_dict["ppo_config_path"]["PPOTrainer_config_path"]
        discriminator_config_path = config_path_dict["discriminator_config_path"]["Discriminator_config_path"]
        reward_weight_config_path = (
            config_path_dict.get("reward_weight_config_path", {}).get("RewardWeight_config_path")
        )
        skill_one_hot_config_path = (
            config_path_dict.get("skill_one_hot_config_path", {}).get("SkillOneHot_config_path")
        )

        seed_from_cli = any(token == "--seed" or token.startswith("--seed=") for token in sys.argv[1:])
        if not seed_from_cli:
            config_seed = _load_seed_from_ppo_config(ppo_config_path)
            if config_seed is not None:
                args.seed = int(config_seed)

        skill_names = _parse_skill_names(args.skill_names)
        ppotrainer_config, ppo_defined_keys = _load_ppo_config(ppo_config_path)
        applied_reward_weight_keys, skipped_reward_weight_keys = _apply_reward_weight_config(
            ppotrainer_config,
            _load_reward_weight_config(reward_weight_config_path),
            skill_names=skill_names,
            protected_keys=ppo_defined_keys,
        )
        ppotrainer_config.device = args.rl_device
        if not getattr(ppotrainer_config, "discriminator_config_path", None):
            ppotrainer_config.discriminator_config_path = discriminator_config_path
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
        print(f"[INFO] Isaac Gym sim device: {args.sim_device}")
        print(f"[INFO] PASIST trainer device: {ppotrainer_config.device}")
        print(f"[INFO] TORCH_EXTENSIONS_DIR: {os.environ.get('TORCH_EXTENSIONS_DIR')}")
        print(
            "[INFO] Episode phase randomization: "
            + ("enabled" if args.randomize_episode_phase else "disabled")
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
        if skill_one_hot_config_path:
            print(f"[INFO] Skill one-hot config: {skill_one_hot_config_path}")
        print(f"[INFO] Seed: {args.seed}")

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

        save_model_dir = pathlib.Path(args.save_model_dir)
        if not save_model_dir.is_absolute():
            save_model_dir = run_dir / save_model_dir

        backend = make_unitree_go2_pasist_backend(
            task_name=args.task,
            num_envs=args.num_envs,
            seed=args.seed,
            sim_device=args.sim_device,
            headless=args.headless,
            use_gpu_pipeline=args.use_gpu_pipeline,
            subscenes=args.subscenes,
            num_threads=args.num_threads,
            disable_noise=not args.enable_noise,
            disable_domain_rand=not args.enable_domain_rand,
            command_resampling_time=args.command_resampling_time,
            randomize_episode_phase=args.randomize_episode_phase,
        )
        env = IsaacGymPasistEnv(
            backend=backend,
            num_envs=args.num_envs,
            skill_names=skill_names,
            target_pose_bank=_load_keyframe_pose_bank(args.target_pose_bank) if args.target_pose_bank else None,
            skill_one_hot_map=_load_skill_one_hot_config(skill_one_hot_config_path),
            policy_obs_key="obs",
            critic_obs_key="states",
            joint_pos_rel_slice=(9, 21),
            root_quat_order="xyzw",
        )

        print(f"[INFO] Log directory: {run_dir}")
        print(f"[INFO] Checkpoint directory: {save_model_dir}")
        print(
            "[INFO] Env policy/critic/action/imitation dims: "
            f"{env.obs_dim}/{env.critic_obs_dim}/{env.action_dim}/{env.imitation_obs_dim}"
        )

        skill_selector = None
        if ppotrainer_config.use_skill_selector:
            skill_selector = SkillSelector(num_skills=env.num_skills, velocity_range=env.velocity_range)

        trainer = PPOTrainer(env=env, config=ppotrainer_config, skill_selector=skill_selector)
        _print_network_overview(trainer=trainer, seed=args.seed)
        run_start_time = time.perf_counter()

        def on_iteration_end(stats: dict[str, float]) -> None:
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
            iteration_num = iteration_index + 1
            if iteration_num % args.save_interval == 0:
                _save_model(trainer, save_model_dir, f"iter_{iteration_num:06d}")

        history = trainer.train(
            num_iterations=args.iterations,
            seed=args.seed,
            on_iteration_end=on_iteration_end,
        )

        _save_model(trainer, save_model_dir, "final")
        with (run_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump([_to_serializable(item) for item in history], file, ensure_ascii=False, indent=2)

        print("[OK] Isaac Gym training finished")
        print(f"[INFO] Training artifacts saved to: {run_dir}")

    except Exception as exc:
        print("[ERROR] Isaac Gym training pipeline failed")
        print(f"type: {type(exc).__name__}")
        print(f"message: {exc}")
        traceback.print_exc()
        raise

    finally:
        if env is not None:
            env.close()
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
