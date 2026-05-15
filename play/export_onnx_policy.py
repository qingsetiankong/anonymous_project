from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import torch
import torch.nn as nn


def _add_project_root_to_path() -> pathlib.Path:
    """确保直接运行脚本时可以导入项目内模块。"""
    project_root = pathlib.Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


PROJECT_ROOT = _add_project_root_to_path()

from rl.ppo_trainer import GaussianPolicy  # noqa: E402


def _torch_load_checkpoint(checkpoint_path: pathlib.Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


class DeterministicDeployPolicy(nn.Module):
    """
    用于部署导出的确定性 actor。

    训练时 `GaussianPolicy` 输出的是高斯分布：
    - `mean_net(obs)` 负责给出动作均值
    - `log_std` 负责训练期采样

    部署到 Isaac Sim / 真实机器人时，一般不再需要采样，
    而是直接使用动作均值作为确定性策略输出。
    """

    def __init__(self, policy: GaussianPolicy) -> None:
        super().__init__()
        self.actor = policy.mean_net

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """输入观测，输出确定性动作。"""
        return self.actor(observations)


def _find_policy_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """
    从 checkpoint 对象中提取 `policy_state_dict`。

    支持：
    - 直接是 `state_dict`
    - 顶层 dict 中包含 `policy_state_dict`
    """
    if isinstance(checkpoint, dict):
        if "policy_state_dict" in checkpoint:
            state_dict = checkpoint["policy_state_dict"]
            if not isinstance(state_dict, dict):
                raise TypeError("checkpoint['policy_state_dict'] 不是一个 state_dict 字典")
            return state_dict

        # 顶层本身就是 state_dict 的场景
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint

    raise KeyError("未在 checkpoint 中找到可用的 policy_state_dict")


def _resolve_actor_activation(checkpoint: object) -> str:
    """
    优先从 checkpoint 中恢复训练时使用的 actor 激活函数。

    如果老 checkpoint 没有保存 config，则回退到 `relu`，以兼容历史导出逻辑。
    """
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict):
            activation = config.get("actor_activation")
            if activation:
                return str(activation)
    return "relu"


def _infer_policy_architecture(state_dict: dict[str, torch.Tensor]) -> tuple[int, tuple[int, ...], int]:
    """
    根据 `mean_net.layers.*.weight` 自动恢复 actor 网络结构。

    返回：
    - obs_dim
    - hidden_dims
    - action_dim
    """
    layer_weights: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        if key.startswith("mean_net.layers.") and key.endswith(".weight"):
            layer_index = int(key.split(".")[2])
            layer_weights.append((layer_index, value))

    if not layer_weights:
        raise KeyError("在 state_dict 中找不到 mean_net.layers.*.weight，无法恢复网络结构")

    layer_weights.sort(key=lambda item: item[0])
    ordered_weights = [weight for _, weight in layer_weights]

    obs_dim = int(ordered_weights[0].shape[1])
    hidden_dims = tuple(int(weight.shape[0]) for weight in ordered_weights[:-1])
    action_dim = int(ordered_weights[-1].shape[0])
    return obs_dim, hidden_dims, action_dim


def _load_policy_from_checkpoint(
    checkpoint_path: pathlib.Path,
    device: torch.device,
) -> tuple[GaussianPolicy, int, tuple[int, ...], int, str]:
    """
    从训练好的 checkpoint 中恢复 `GaussianPolicy`。
    """
    checkpoint = _torch_load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    state_dict = _find_policy_state_dict(checkpoint)
    obs_dim, hidden_dims, action_dim = _infer_policy_architecture(state_dict)
    actor_activation = _resolve_actor_activation(checkpoint)

    policy = GaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        init_log_std=-0.5,  # load_state_dict 后会被 checkpoint 中真实参数覆盖
        hidden_activation=actor_activation,
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, obs_dim, hidden_dims, action_dim, actor_activation


def load_policy_from_checkpoint(
    checkpoint_path: pathlib.Path,
    device: torch.device,
) -> tuple[GaussianPolicy, int, tuple[int, ...], int, str]:
    """公开的 checkpoint 策略恢复接口，供导出和 play 共用。"""
    return _load_policy_from_checkpoint(checkpoint_path=checkpoint_path, device=device)


def _format_hidden_dims(hidden_dims: Iterable[int]) -> str:
    """把隐藏层维度格式化成便于打印的字符串。"""
    dims = list(hidden_dims)
    return "(" + ", ".join(str(dim) for dim in dims) + ")"


def export_onnx_policy(
    checkpoint_path: pathlib.Path,
    output_path: pathlib.Path,
    opset_version: int = 17,
    device: str = "cpu",
) -> pathlib.Path:
    """
    将训练得到的 `policy_state_dict` 导出为部署用的 ONNX actor。

    注意：
    - 导出的是确定性策略，即 `mean_net(obs)`
    - 不包含 value function
    - 不包含 discriminator
    - 不包含 obs normalizer，因为当前项目训练链本身没有启用 obs normalization
    """
    torch_device = torch.device(device)
    policy, obs_dim, hidden_dims, action_dim, actor_activation = load_policy_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch_device,
    )
    deploy_policy = DeterministicDeployPolicy(policy).to(torch_device)
    deploy_policy.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32, device=torch_device)

    torch.onnx.export(
        deploy_policy,
        dummy_obs,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={
            "observations": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    )

    print("[INFO] ONNX export summary")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output: {output_path}")
    print(f"obs_dim: {obs_dim}")
    print(f"action_dim: {action_dim}")
    print(f"hidden_dims: {_format_hidden_dims(hidden_dims)}")
    print(f"actor_activation: {actor_activation}")
    print(f"opset_version: {opset_version}")
    print("[INFO] Exported deterministic actor: actions = mean_net(observations)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="导出当前项目训练得到的 policy 为 ONNX 文件。")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="result/final.pt",
        help="训练得到的 checkpoint 路径。默认读取 result/final.pt。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="导出的 ONNX 文件路径。默认保存在 checkpoint 同目录下的 policy.onnx。",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset 版本。默认 17。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="导出时使用的 torch device。建议保持 cpu。",
    )
    args = parser.parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

    if args.output is None:
        output_path = checkpoint_path.with_name("policy.onnx")
    else:
        output_path = pathlib.Path(args.output).expanduser().resolve()

    export_onnx_policy(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        opset_version=int(args.opset),
        device=args.device,
    )


if __name__ == "__main__":
    main()
