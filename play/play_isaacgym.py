"""
Play script for old Pasist checkpoints (logs_isaacgym/...).

Loads the old [128,128,128] network and runs with a PasistLeggedRobot env.

Usage:
    python play/play_isaacgym.py --checkpoint logs_isaacgym/go2/20260513-164116/checkpoints/iter_000100.pt
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
import traceback


def _add_project_root_to_path() -> pathlib.Path:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
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

    parser = argparse.ArgumentParser(description="Play old Pasist checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=-1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--cmd-x", type=float, default=0.0,
                        help="Fixed x velocity. Set to 0 with --cmd-resample to use random")
    parser.add_argument("--cmd-y", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--cmd-resample", type=float, default=10.0,
                        help="Command resample interval (s). 0=use fixed cmd, >0=random")
    parser.add_argument("--enable-noise", action="store_true", default=False)
    args = parser.parse_args()

    env = None

    try:
        import isaacgym  # noqa: F401
        import legged_gym.envs  # noqa: F401
        from isaacgym import gymapi

        import numpy as np
        import torch

        from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed

        from envs.pasist.pasist_robot import PasistLeggedRobot
        from envs.pasist.pasist_robot_config import PasistRobotCfg

        checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Env config
        env_cfg = PasistRobotCfg()
        env_cfg.env.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.noise.add_noise = args.enable_noise
        env_cfg.commands.resampling_time = args.cmd_resample if args.cmd_resample > 0 else 1e9
        env_cfg.commands.curriculum = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.push_robots = False
        _cmd_resample = args.cmd_resample > 0

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

        print("[INFO] Creating env...")
        env = PasistLeggedRobot(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device=args.sim_device,
            headless=args.headless,
            obs_history_len=0,
        )
        print(f"[INFO] obs_dim={env.num_obs}, action_dim={env.num_actions}")

        # Load old checkpoint (separate policy/value state dicts)
        print(f"[INFO] Loading: {checkpoint_path}")
        policy_device = torch.device(args.policy_device)
        ckpt = torch.load(checkpoint_path, map_location=policy_device)

        policy_state = ckpt["policy_state_dict"]
        value_state = ckpt["value_state_dict"]
        old_config = ckpt.get("config", {})

        # Infer architecture from checkpoint shapes
        # policy: mean_net.layers.0.weight [128, 49] -> 3 hidden layers of 128 -> output [12, 128]
        # value:   value_net.layers.0.weight [128, 52] -> 3 hidden layers of 128 -> output [1, 128]
        actor_obs_dim = policy_state["mean_net.layers.0.weight"].shape[1]   # 49
        critic_obs_dim = value_state["value_net.layers.0.weight"].shape[1]   # 52
        action_dim = policy_state["mean_net.layers.3.weight"].shape[0]       # 12

        # Hidden dims: from layers 0→1 weight shape
        hidden_0 = policy_state["mean_net.layers.0.weight"].shape[0]  # 128
        hidden_1 = policy_state["mean_net.layers.1.weight"].shape[0]  # 128
        hidden_2 = policy_state["mean_net.layers.2.weight"].shape[0]  # 128
        actor_hidden = (hidden_0, hidden_1, hidden_2)

        from rl.pasist_actor_critic import PasistActorCritic

        # Checkpoint uses "log_std" (log-scale), model uses "std_param" (direct scale)
        if "log_std" in policy_state:
            policy_state["std_param"] = torch.exp(policy_state.pop("log_std"))

        model = PasistActorCritic(
            num_actor_obs=actor_obs_dim,
            num_critic_obs=critic_obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden,
            critic_hidden_dims=actor_hidden,
        )
        model.actor.load_state_dict(policy_state)
        model.critic.load_state_dict(value_state)
        model.to(policy_device)
        model.eval()

        actor_params = sum(p.numel() for p in model.actor.parameters())
        ckpt_iter = old_config.get("iter", checkpoint_path.stem)
        print(f"[INFO] Loaded: iter={ckpt_iter}")
        print(f"[INFO]   actor: [{actor_obs_dim}→{hidden_0}→{hidden_1}→{hidden_2}→{action_dim}] {actor_params:,} params")

        # Set initial command
        env.commands[:, 0] = args.cmd_x
        env.commands[:, 1] = args.cmd_y
        env.commands[:, 2] = args.cmd_yaw

        obs_dict = env.get_observations()
        obs = obs_dict.to(policy_device)

        step_dt = env.dt * env.cfg.control.decimation
        if _cmd_resample:
            print(f"[INFO] cmd_resample={args.cmd_resample}s  (random, env-managed)")
        else:
            print(f"[INFO] cmd=[{args.cmd_x}, {args.cmd_y}, {args.cmd_yaw}]  (fixed)")
        print(f"[INFO] step_dt: {step_dt:.4f}s")
        print(f"[INFO] Press Ctrl+C to stop\n")

        step_index = 0
        while args.num_steps < 0 or step_index < args.num_steps:
            loop_start = time.perf_counter()

            with torch.inference_mode():
                actions = model.actor.mean_net(obs)

            ret = env.step(actions)
            obs, _, rewards, dones, _ = ret
            obs = obs.to(policy_device)

            # Only fix command when resample is disabled
            if not _cmd_resample:
                env.commands[:, 0] = args.cmd_x
                env.commands[:, 1] = args.cmd_y
                env.commands[:, 2] = args.cmd_yaw

            if step_index == 0 or (
                args.log_interval > 0 and (step_index + 1) % args.log_interval == 0
            ):
                r = rewards.mean().item()
                print(f"[{step_index + 1:5d}] reward={r:.4f}  done={int(dones.sum().item())}")

            step_index += 1

            if args.real_time:
                elapsed = time.perf_counter() - loop_start
                sleep_time = step_dt - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        print(f"[OK] Finished after {step_index} steps")

    except KeyboardInterrupt:
        print(f"\n[OK] Interrupted after {step_index} steps")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            try:
                if hasattr(env, "gym") and hasattr(env, "sim"):
                    viewer = getattr(env, "viewer", None)
                    if viewer is not None:
                        env.gym.destroy_viewer(viewer)
                    env.gym.destroy_sim(env.sim)
            except Exception:
                pass


if __name__ == "__main__":
    main()
