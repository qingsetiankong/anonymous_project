"""
Play script for Phase 1 checkpoints.

Loads a PasistLeggedRobot env and a saved PasistActorCritic checkpoint,
then runs the policy with the viewer open.

Usage:
    python play/play_phase1.py --checkpoint logs_phase1/20260513_XXXX_phase1/model_100.pt
    python play/play_phase1.py --checkpoint logs_phase1/20260513_XXXX_phase1/model_100.pt --command-velocity 0.5
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

    parser = argparse.ArgumentParser(description="Play Phase 1 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--task", type=str, default="go2")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--command-velocity-x", type=float, default=0.3,
                        help="Fixed x-velocity command (forward speed)")
    parser.add_argument("--command-velocity-y", type=float, default=0.0,
                        help="Fixed y-velocity command (lateral speed)")
    parser.add_argument("--command-yaw", type=float, default=0.0,
                        help="Fixed yaw angular velocity command (turning speed)")
    parser.add_argument("--random-commands", action="store_true",
                        help="Use random commands each resample instead of fixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=-1,
                        help="Number of steps (-1 = run forever)")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--real-time", action="store_true",
                        help="Sleep to match real-time physics")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run without viewer")
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

        # Create env config
        env_cfg = PasistRobotCfg()
        env_cfg.env.num_envs = args.num_envs
        env_cfg.seed = args.seed

        # Create env with viewer
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

        print("[INFO] Creating PasistLeggedRobot env...")
        env = PasistLeggedRobot(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device=args.sim_device,
            headless=args.headless,
            obs_history_len=0,
        )
        print(f"[INFO] Env created: {env.num_envs} envs, "
              f"obs_dim={env.num_obs}, action_dim={env.num_actions}")

        # Load checkpoint
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        policy_device = torch.device(args.policy_device)
        checkpoint = torch.load(checkpoint_path, map_location=policy_device)

        from rl.pasist_actor_critic import PasistActorCritic

        # Model architecture must match the training config: [256, 256]
        actor_critic = PasistActorCritic(
            num_actor_obs=env.num_obs,
            num_critic_obs=env.num_privileged_obs,
            num_actions=env.num_actions,
            actor_hidden_dims=(256, 256),
            critic_hidden_dims=(256, 256),
        )
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        actor_critic.to(policy_device)
        actor_critic.eval()

        ckpt_iter = checkpoint.get("iter", "?")
        print(f"[INFO] Checkpoint loaded: iter={ckpt_iter}")
        print(f"[INFO]   actor params: {sum(p.numel() for p in actor_critic.actor.parameters()):,}")
        print(f"[INFO]   critic params: {sum(p.numel() for p in actor_critic.critic.parameters()):,}")

        # Reset env
        obs_dict = env.get_observations()
        obs = obs_dict.to(policy_device)

        step_dt = env.dt * env.cfg.control.decimation

        print(f"[INFO] step_dt: {step_dt:.4f}s")
        print(f"[INFO] Commands are randomly resampled by the env every 10s")
        print(f"[INFO] Press Ctrl+C to stop")

        step_index = 0
        while args.num_steps < 0 or step_index < args.num_steps:
            loop_start = time.perf_counter()

            with torch.inference_mode():
                actions = actor_critic.actor.mean_net(obs)

            # Step env
            ret = env.step(actions)
            obs, _, rewards, dones, _ = ret
            obs = obs.to(policy_device)

            if step_index == 0 or (
                args.log_interval > 0 and (step_index + 1) % args.log_interval == 0
            ):
                r = rewards.mean().item()
                print(
                    f"[INFO] step={step_index + 1:5d}  "
                    f"reward={r:.4f}  "
                    f"done_count={int(dones.sum().item())}"
                )

            step_index += 1

            if args.real_time:
                elapsed = time.perf_counter() - loop_start
                sleep_time = step_dt - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        print(f"[OK] Play finished after {step_index} steps")

    except KeyboardInterrupt:
        print(f"\n[OK] Interrupted after {step_index} steps")
    except Exception as exc:
        print("[ERROR] Play failed")
        print(f"type: {type(exc).__name__}")
        print(f"message: {exc}")
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
