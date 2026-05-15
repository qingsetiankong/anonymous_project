from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
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


def _to_numpy(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _quat_xyzw_to_wxyz(quaternion_xyzw):
    import numpy as np

    quat = np.asarray(quaternion_xyzw, dtype=np.float32).reshape(4)
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)


def _compare_arrays(label: str, left, right) -> tuple[float, float]:
    import numpy as np

    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    diff = np.abs(left_array - right_array)
    max_abs = float(diff.max()) if diff.size > 0 else 0.0
    mean_abs = float(diff.mean()) if diff.size > 0 else 0.0
    print(f"[INFO] {label}: max_abs_diff={max_abs:.8f}, mean_abs_diff={mean_abs:.8f}")
    return max_abs, mean_abs


def _run_onnx_inference(
    model_path: pathlib.Path,
    observations,
    input_name: str,
    output_name: str,
    onnxruntime_python: str | None,
):
    import importlib.util
    import numpy as np

    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        available_inputs = [item.name for item in session.get_inputs()]
        available_outputs = [item.name for item in session.get_outputs()]
        resolved_input_name = input_name if input_name in available_inputs else available_inputs[0]
        resolved_output_name = output_name if output_name in available_outputs else available_outputs[0]
        return np.asarray(
            session.run([resolved_output_name], {resolved_input_name: np.asarray(observations, dtype=np.float32)})[0],
            dtype=np.float32,
        )

    if not onnxruntime_python:
        raise ModuleNotFoundError(
            "当前 Python 环境没有 onnxruntime。请传 `--onnxruntime-python` 指向装有 onnxruntime 的解释器。"
        )

    helper_script = pathlib.Path(__file__).resolve().parent / "onnx_batch_infer.py"
    with tempfile.TemporaryDirectory(prefix="deploy_onnx_eval_") as temp_dir:
        temp_root = pathlib.Path(temp_dir)
        input_path = temp_root / "observations.npy"
        output_path = temp_root / "actions.npy"
        np.save(input_path, np.asarray(observations, dtype=np.float32))
        subprocess.run(
            [
                onnxruntime_python,
                str(helper_script),
                "--model",
                str(model_path),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--input-name",
                input_name,
                "--output-name",
                output_name,
            ],
            check=True,
        )
        return np.load(output_path).astype(np.float32)


def _export_onnx_with_fallback(
    checkpoint_path: pathlib.Path,
    output_path: pathlib.Path,
    onnx_python: str | None,
) -> pathlib.Path:
    if importlib.util.find_spec("onnx") is not None:
        from play.export_onnx_policy import export_onnx_policy

        return export_onnx_policy(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            device="cpu",
        )

    if not onnx_python:
        raise ModuleNotFoundError(
            "当前 Python 环境没有 onnx。请传 `--onnxruntime-python` 指向同时装有 onnx 和 onnxruntime 的解释器。"
        )

    export_script = pathlib.Path(__file__).resolve().parents[1] / "play" / "export_onnx_policy.py"
    subprocess.run(
        [
            onnx_python,
            str(export_script),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ],
        check=True,
    )
    return output_path


def main() -> None:
    project_root = _add_project_root_to_path()
    _prepare_isaacgym_runtime()

    from numpy_compat import ensure_numpy_legacy_aliases

    ensure_numpy_legacy_aliases()

    parser = argparse.ArgumentParser(description="离线验证 checkpoint / ONNX / deploy 观测重建是否一致。")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--deploy-config", type=str, default="deploy/deploy.yaml")
    parser.add_argument("--onnx-output", type=str, default=None)
    parser.add_argument(
        "--onnxruntime-python",
        type=str,
        default=None,
        help="可选外部解释器路径。当前环境缺少 onnx/onnxruntime 时，会借它完成 ONNX 导出和推理。",
    )
    parser.add_argument("--task", type=str, default="go2")
    parser.add_argument("--skill-names", type=str, default="walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--sim-device", type=str, default="cpu")
    parser.add_argument("--policy-device", type=str, default="cpu")
    parser.add_argument("--disable-gpu-pipeline", dest="use_gpu_pipeline", action="store_false")
    parser.set_defaults(use_gpu_pipeline=True)
    parser.add_argument("--target-pose-bank", type=str, default="target_pose_bank/target_pose_bank.npy")
    parser.add_argument("--config-path", type=str, default="configs/config_path.yaml")
    args = parser.parse_args()

    env = None

    try:
        import isaacgym  # noqa: F401
        import numpy as np
        import torch

        from deploy.go2_controller import (
            build_policy_observation_from_state,
            load_deploy_config,
        )
        from envs.IsaacGymPasistEnv import IsaacGymPasistEnv
        from envs.isaacgym_unitree_go2_backend import make_unitree_go2_pasist_backend
        from play.export_onnx_policy import load_policy_from_checkpoint
        from training.train import (
            _load_config_path,
            _load_keyframe_pose_bank,
            _load_skill_one_hot_config,
            _parse_skill_names,
        )

        checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

        deploy_config_path = pathlib.Path(args.deploy_config).expanduser().resolve()
        if not deploy_config_path.exists():
            raise FileNotFoundError(f"找不到 deploy config: {deploy_config_path}")

        deploy_cfg = load_deploy_config(deploy_config_path)
        config_path_dict = _load_config_path(args.config_path)
        skill_one_hot_config_path = (
            config_path_dict.get("skill_one_hot_config_path", {}).get("SkillOneHot_config_path")
        )
        skill_names = _parse_skill_names(args.skill_names)

        backend = make_unitree_go2_pasist_backend(
            task_name=args.task,
            num_envs=args.num_envs,
            seed=args.seed,
            sim_device=args.sim_device,
            headless=True,
            use_gpu_pipeline=args.use_gpu_pipeline,
            disable_noise=True,
            disable_domain_rand=True,
            command_resampling_time=1.0e9,
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

        if deploy_cfg.policy_obs_dim != env.obs_dim:
            raise ValueError(
                "deploy 配置重建的 policy obs dim 与训练环境不一致："
                f" deploy={deploy_cfg.policy_obs_dim}, env={env.obs_dim}"
            )

        policy_device = torch.device(args.policy_device)
        policy, obs_dim, _, action_dim, actor_activation = load_policy_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=policy_device,
        )
        if obs_dim != env.obs_dim or action_dim != env.action_dim:
            raise ValueError(
                "checkpoint 网络维度与当前环境不一致："
                f" checkpoint_obs={obs_dim}, env_obs={env.obs_dim},"
                f" checkpoint_act={action_dim}, env_act={env.action_dim}"
            )

        if args.onnx_output is None:
            onnx_output_path = checkpoint_path.with_name("policy.onnx")
        else:
            onnx_output_path = pathlib.Path(args.onnx_output).expanduser().resolve()
        _export_onnx_with_fallback(
            checkpoint_path=checkpoint_path,
            output_path=onnx_output_path,
            onnx_python=args.onnxruntime_python,
        )

        command = env.build_command(
            velocity=float(deploy_cfg.fixed_velocity_command[0]),
            skill_id=int(deploy_cfg.command_skill_id),
        )
        observation, _ = env.reset(command=command, seed=args.seed)

        observation_batch_list: list[np.ndarray] = []
        torch_action_list: list[np.ndarray] = []
        obs_diff_max = 0.0
        obs_diff_mean = 0.0
        last_action = np.zeros((env.action_dim,), dtype=np.float32)

        for step_index in range(int(args.num_steps)):
            joint_pos_policy = _to_numpy(backend.dof_pos)[0]
            joint_vel_policy = _to_numpy(backend.dof_vel)[0]
            base_quat_xyzw = _to_numpy(backend.inner_env.base_quat)[0]
            imu_quaternion_wxyz = _quat_xyzw_to_wxyz(base_quat_xyzw)
            imu_gyro = _to_numpy(backend.inner_env.base_ang_vel)[0]

            deploy_observation = build_policy_observation_from_state(
                cfg=deploy_cfg,
                joint_pos_policy=joint_pos_policy,
                joint_vel_policy=joint_vel_policy,
                imu_quaternion_wxyz=imu_quaternion_wxyz,
                imu_gyro=imu_gyro,
                last_action=last_action,
            )
            current_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            max_abs, mean_abs = _compare_arrays(
                label=f"deploy_obs_vs_env_obs_step_{step_index}",
                left=deploy_observation,
                right=current_observation,
            )
            obs_diff_max = max(obs_diff_max, max_abs)
            obs_diff_mean = max(obs_diff_mean, mean_abs)

            obs_tensor = torch.as_tensor(
                deploy_observation.reshape(1, -1),
                dtype=torch.float32,
                device=policy_device,
            )
            with torch.inference_mode():
                action_tensor = policy.mean_net(obs_tensor)
            action = action_tensor.detach().cpu().numpy().astype(np.float32)[0]

            observation_batch_list.append(deploy_observation.astype(np.float32))
            torch_action_list.append(action.astype(np.float32))

            observation, _, terminated, truncated, _ = env.step(action)
            done = bool(np.asarray(terminated).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0])
            last_action = action
            if done:
                observation, _ = env.reset(command=command, seed=args.seed + step_index + 1)
                last_action = np.zeros((env.action_dim,), dtype=np.float32)

        observation_batch = np.asarray(observation_batch_list, dtype=np.float32)
        torch_action_batch = np.asarray(torch_action_list, dtype=np.float32)
        onnx_action_batch = _run_onnx_inference(
            model_path=onnx_output_path,
            observations=observation_batch,
            input_name="observations",
            output_name="actions",
            onnxruntime_python=args.onnxruntime_python,
        )
        onnx_diff_max, onnx_diff_mean = _compare_arrays(
            label="onnx_action_vs_torch_action",
            left=onnx_action_batch,
            right=torch_action_batch,
        )

        print("[INFO] Validation summary")
        print(f"checkpoint: {checkpoint_path}")
        print(f"onnx_output: {onnx_output_path}")
        print(f"actor_activation: {actor_activation}")
        print(f"deploy_obs_dim: {deploy_cfg.policy_obs_dim}")
        print(f"env_obs_dim: {env.obs_dim}")
        print(f"max deploy obs diff: {obs_diff_max:.8f}")
        print(f"max onnx action diff: {onnx_diff_max:.8f}")

        if obs_diff_max > 1.0e-4:
            raise AssertionError(
                "deploy 观测重建与训练环境 observation 不一致。"
                f" max_abs_diff={obs_diff_max:.8f}"
            )
        if onnx_diff_max > 1.0e-4:
            raise AssertionError(
                "ONNX 输出与 PyTorch actor 输出不一致。"
                f" max_abs_diff={onnx_diff_max:.8f}"
            )

        print("[OK] Deploy policy validation passed")

    except Exception as exc:
        print("[ERROR] Deploy policy validation failed")
        print(f"type: {type(exc).__name__}")
        print(f"message: {exc}")
        traceback.print_exc()
        raise

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
