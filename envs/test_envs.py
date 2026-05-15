from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any


def _add_project_root_to_path() -> pathlib.Path:
    """
    确保从 envs/test_envs.py 直接运行时，也能 import 项目内的 envs 包。

    例如：
    python envs/test_envs.py

    这种运行方式下，Python 默认会把 envs/ 放到 sys.path[0]，
    但不会自动把项目根目录放进去，所以这里手动补上。
    """
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def _summarize_value(value: Any) -> str:
    """
    打印 observation / reward / done 时使用的简短摘要函数。

    Isaac Lab 的 reset/step 返回值可能是：
    - torch.Tensor
    - dict[str, torch.Tensor]
    - tuple / list
    - 普通标量

    为了测试脚本通用，这里只打印 type、shape、dtype 等关键信息。
    """
    if isinstance(value, dict):
        items = ", ".join(f"{key}: {_summarize_value(item)}" for key, item in value.items())
        return "{" + items + "}"
    if isinstance(value, (tuple, list)):
        items = ", ".join(_summarize_value(item) for item in value)
        return f"{type(value).__name__}({items})"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None:
        return f"{type(value).__name__}(shape={tuple(shape)}, dtype={dtype}, device={device})"
    return f"{type(value).__name__}({value})"


def _sample_random_action(env):
    """
    为 Isaac Lab 环境构造随机动作。

    不直接使用 env.action_space.sample() 的原因：
    - Gym action_space.sample() 返回 numpy.ndarray
    - Isaac Lab 的 ManagerBasedRLEnv 内部通常使用 torch.Tensor
    - 直接传 numpy 有时会导致额外转换或后端阻塞，不利于测试定位
    """
    import torch

    unwrapped = env.unwrapped
    num_envs = unwrapped.num_envs
    device = unwrapped.device

    # action_manager.total_action_dim 是 Isaac Lab 中真实 action 维度。
    # 如果某些包装环境没有这个属性，再回退到 action_space.shape。
    action_dim = getattr(unwrapped.action_manager, "total_action_dim", None)
    if action_dim is None:
        if not hasattr(env, "action_space") or len(env.action_space.shape) == 0:
            raise AttributeError("无法从 action_manager 或 action_space 推断动作维度")
        action_dim = env.action_space.shape[-1]

    # 最小测试用小幅随机动作，避免一上来给非常大的随机关节目标导致机器人瞬间崩溃。
    return 0.1 * torch.randn(num_envs, action_dim, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查项目中的 Isaac Lab 环境是否可创建、reset 和 step。")
    #parser.add_argument("--show-gui", action="store_true", help="显示 Isaac Sim GUI；默认使用 headless 模式测试。")
    parser.add_argument("--num-envs", type=int, default=4, help="测试时覆盖并行环境数量。")
    parser.add_argument("--steps", type=int, default=3, help="reset 后随机 step 的步数。")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="要测试的 Gym task id；默认使用 envs.isaacsim_mini_envs.TASK_ID。",
    )

    # AppLauncher.add_app_launcher_args 会添加 --device 等 Isaac Lab 常用参数。
    # 必须在启动 Isaac Sim 之前完成参数解析。
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    _add_project_root_to_path()

    # 关键点：先启动 Isaac Sim App，再 import 含 isaaclab.sim / unitree asset cfg 的环境配置。
    # 这样 omni.client 等 Omniverse 模块才有机会被正确加载。
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym

        import envs.isaacsim_mini_envs as mini_envs
        import envs.pasist_env_cfg as pasist_envs

        task_id = args.task or mini_envs.TASK_ID
        print(f"[INFO] Testing task: {task_id}")

        # 如果测试的是默认最小环境，直接实例化它的 cfg 并覆盖 num_envs。
        # 如果传入其它 task，则尝试从 Gym 注册信息里使用 env_cfg_entry_point。
        if task_id == mini_envs.TASK_ID:
            env_cfg = mini_envs.Go2MiniEnvCfg()
            env_cfg.scene.num_envs = args.num_envs
        elif task_id == pasist_envs.TASK_ID:
            env_cfg = pasist_envs.Go2PasistEnvCfg()
            env_cfg.scene.num_envs = args.num_envs
        else:
            spec = gym.spec(task_id)
            env_cfg_entry_point = spec.kwargs.get("env_cfg_entry_point")
            if env_cfg_entry_point is None:
                raise KeyError(f"{task_id} 没有 env_cfg_entry_point，无法自动创建 env cfg")

            module_name, class_name = env_cfg_entry_point.split(":")
            module = __import__(module_name, fromlist=[class_name])
            env_cfg = getattr(module, class_name)() 
            if hasattr(env_cfg, "scene"):
                env_cfg.scene.num_envs = args.num_envs

        env = gym.make(task_id, cfg=env_cfg)
        print("[OK] gym.make succeeded")
        print(f"[INFO] action_space: {env.action_space}")
        print(f"[INFO] observation_space: {env.observation_space}")

        reset_result = env.reset()
        print("[OK] env.reset succeeded")
        print(f"[INFO] reset result: {_summarize_value(reset_result)}")

        for step_index in range(args.steps):
            print(f"[INFO] Sampling action for step {step_index + 1}/{args.steps}")
            action = _sample_random_action(env)
            print(f"[INFO] action: {_summarize_value(action)}")
            print(f"[INFO] Calling env.step for step {step_index + 1}/{args.steps}")
            step_result = env.step(action)
            print(f"[OK] env.step {step_index + 1}/{args.steps} succeeded")
            print(f"[INFO] step result: {_summarize_value(step_result)}")

        env.close()
        print("[OK] Environment test finished")

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
