from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from datetime import datetime


def _add_project_root_to_path() -> pathlib.Path:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # also add unitree_rl_gym for legged_gym imports
    legged_gym_root = pathlib.Path("/media/ubuntu20/D/NvidiaIsaac/unitree_rl_gym")
    if legged_gym_root.exists() and str(legged_gym_root) not in sys.path:
        sys.path.insert(0, str(legged_gym_root))
    return project_root


def _prepare_runtime() -> None:
    python_bin = pathlib.Path(sys.executable).resolve().parent
    path = os.environ.get("PATH", "")
    if str(python_bin) not in path.split(os.pathsep):
        os.environ["PATH"] = str(python_bin) + os.pathsep + path
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")


def main() -> None:
    project_root = _add_project_root_to_path()
    _prepare_runtime()

    from numpy_compat import ensure_numpy_legacy_aliases

    ensure_numpy_legacy_aliases()

    parser = argparse.ArgumentParser(description="Pasist Phase 1: legged_gym locomotion baseline")
    parser.add_argument("--task", type=str, default="go2")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--rl-device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--log-dir", type=str, default="logs_phase1")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--history-len", type=int, default=0,
                        help="Number of history frames for observation stacking")
    args = parser.parse_args()

    import isaacgym  # noqa: F401
    from isaacgym import gymapi

    # legged_gym has a circular import between envs/__init__.py and
    # utils/task_registry.py.  Importing legged_gym.envs first resolves it.
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
    from legged_gym.utils.task_registry import task_registry

    from envs.pasist.pasist_robot import PasistLeggedRobot
    from envs.pasist.pasist_robot_config import PasistRobotCfg, PasistRobotCfgPPO
    from runner.on_policy_runner import OnPolicyRunner

    # --- register task ---
    task_registry.register(
        "go2_phase1", PasistLeggedRobot, PasistRobotCfg(), PasistRobotCfgPPO()
    )

    env_cfg, train_cfg_obj = task_registry.get_cfgs("go2_phase1")
    train_cfg_dict = class_to_dict(train_cfg_obj)

    # override with CLI args
    env_cfg.env.num_envs = args.num_envs
    env_cfg.seed = args.seed
    train_cfg_dict["runner"]["max_iterations"] = args.iterations

    # --- create env ---
    set_seed(env_cfg.seed)
    sim_params = parse_sim_params(
        type("Args", (), {
            "sim_device": args.sim_device,
            "device": args.sim_device,
            "headless": args.headless,
            "use_gpu": args.sim_device.startswith("cuda"),
            "use_gpu_pipeline": True,
            "physics_engine": gymapi.SIM_PHYSX,
            "subscenes": 0,
            "num_threads": 10,
        }),
        {"sim": class_to_dict(env_cfg.sim)},
    )

    env = PasistLeggedRobot(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=args.sim_device,
        headless=args.headless,
        obs_history_len=args.history_len,
    )
    print(f"[INFO] PasistLeggedRobot created: {env.num_envs} envs, "
          f"obs_dim={env.num_obs}, privileged_obs_dim={env.num_privileged_obs}, "
          f"action_dim={env.num_actions}, history_len={args.history_len}")

    # --- setup logging ---
    log_root = pathlib.Path(args.log_dir)
    if not log_root.is_absolute():
        log_root = project_root / log_root
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = log_root / f"{timestamp}_phase1"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Log directory: {run_dir}")

    # --- create runner ---
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg_dict,
        log_dir=str(run_dir),
        device=args.rl_device,
    )

    print(f"[INFO] Starting Phase 1 training: {args.iterations} iterations")
    print(f"[INFO] num_steps_per_env={runner.num_steps_per_env}, "
          f"num_envs={env.num_envs}, "
          f"steps_per_iteration={runner.num_steps_per_env * env.num_envs}")

    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=True,
    )

    print("[OK] Phase 1 training finished")
    print(f"[INFO] Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
