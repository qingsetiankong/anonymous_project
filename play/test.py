import os
import torch
from collections.abc import Mapping, Sequence

def print_divider(title=None, width=80):
    if title is None:
        print("=" * width)
    else:
        text = f" {title} "
        pad = max(0, width - len(text))
        left = pad // 2
        right = pad - left
        print("=" * left + text + "=" * right)

def safe_type_name(obj):
    try:
        return type(obj).__name__
    except Exception:
        return str(type(obj))

def summarize_tensor(tensor, name="", max_stats=True):
    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    device = tensor.device
    print(f"{name}: Tensor, shape={shape}, dtype={dtype}, device={device}")
    if max_stats:
        try:
            t = tensor.detach().float().cpu()
            print(
                f"    stats -> min={t.min().item():.6f}, max={t.max().item():.6f}, "
                f"mean={t.mean().item():.6f}, std={t.std().item():.6f}"
            )
        except Exception as e:
            print(f"    stats -> unavailable ({e})")

def summarize_state_dict(state_dict, title="state_dict", max_items=200):
    print_divider(title)
    if not isinstance(state_dict, Mapping):
        print(f"Not a mapping, actual type: {safe_type_name(state_dict)}")
        return

    print(f"Number of parameters/buffers: {len(state_dict)}")

    total_params = 0
    shown = 0
    linear_like = []

    for k, v in state_dict.items():
        if torch.is_tensor(v):
            numel = v.numel()
            total_params += numel
            print(f"{k:60s} shape={tuple(v.shape)!s:20s} dtype={str(v.dtype):15s} numel={numel}")
            # 尝试推测线性层
            if v.ndim == 2:
                linear_like.append((k, tuple(v.shape)))
        else:
            print(f"{k:60s} type={safe_type_name(v)}")

        shown += 1
        if shown >= max_items:
            if len(state_dict) > max_items:
                print(f"... truncated, showing first {max_items} items only")
            break

    print(f"\nEstimated total tensor elements: {total_params}")

    if linear_like:
        print_divider("Possible linear layer weights (2D tensors)")
        for name, shape in linear_like:
            print(f"{name:60s} {shape}")

def inspect_object(obj, name="root", depth=0, max_depth=3, max_items=20):
    indent = "  " * depth
    tname = safe_type_name(obj)

    if torch.is_tensor(obj):
        print(f"{indent}{name}: Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device}")
        return

    if isinstance(obj, Mapping):
        print(f"{indent}{name}: dict-like ({tname}), len={len(obj)}")
        if depth >= max_depth:
            return
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{indent}  ... truncated after {max_items} items")
                break
            inspect_object(v, name=str(k), depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        print(f"{indent}{name}: sequence ({tname}), len={len(obj)}")
        if depth >= max_depth:
            return
        for i, v in enumerate(obj[:max_items]):
            inspect_object(v, name=f"[{i}]", depth=depth + 1, max_depth=max_depth, max_items=max_items)
        if len(obj) > max_items:
            print(f"{indent}  ... truncated after {max_items} items")
        return

    print(f"{indent}{name}: {tname}")
    # 可选：打印基础类型的值
    if isinstance(obj, (int, float, bool, str)):
        print(f"{indent}  value={obj}")

def guess_useful_keys(ckpt):
    print_divider("Guessed useful keys")
    if not isinstance(ckpt, Mapping):
        print("Checkpoint is not dict-like, cannot guess keys.")
        return

    candidates = []
    patterns = [
        "policy",
        "actor",
        "critic",
        "model",
        "state_dict",
        "optimizer",
        "normalizer",
        "obs",
        "running_mean",
        "running_var",
        "mean",
        "var",
    ]

    for k in ckpt.keys():
        kl = str(k).lower()
        if any(p in kl for p in patterns):
            candidates.append(k)

    if not candidates:
        print("No obvious useful keys found by name pattern.")
    else:
        for k in candidates:
            print(f"- {k}")

def inspect_checkpoint(path):
    print_divider("Load checkpoint")
    print(f"Path: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # 尝试优先用 weights_only=False 兼容老checkpoint结构
    try:
        ckpt = torch.load(path, map_location="cpu")
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    print(f"Top-level object type: {safe_type_name(ckpt)}")

    print_divider("Top-level structure")
    inspect_object(ckpt, name="checkpoint", max_depth=2, max_items=30)

    if isinstance(ckpt, Mapping):
        print_divider("Top-level keys")
        for k in ckpt.keys():
            print(f"- {k}")

        guess_useful_keys(ckpt)

        # 常见 key 优先检查
        common_state_dict_keys = [
            "policy_state_dict",
            "model_state_dict",
            "state_dict",
            "actor_state_dict",
            "critic_state_dict",
            "actor_critic_state_dict",
        ]

        found_any = False
        for key in common_state_dict_keys:
            if key in ckpt:
                found_any = True
                summarize_state_dict(ckpt[key], title=f"Contents of {key}")

        # 如果没有这些常见 key，但顶层自己就是一个 state_dict 风格结构
        if not found_any:
            all_tensor_like = True
            for _, v in ckpt.items():
                if not torch.is_tensor(v):
                    all_tensor_like = False
                    break
            if all_tensor_like:
                summarize_state_dict(ckpt, title="Top-level seems to be a state_dict")

        # 检查可能的 normalizer / obs rms
        possible_norm_keys = [
            "normalizer",
            "obs_normalizer",
            "obs_rms",
            "running_mean_std",
            "running_mean_std_state_dict",
        ]
        for key in possible_norm_keys:
            if key in ckpt:
                print_divider(f"Possible normalizer info: {key}")
                inspect_object(ckpt[key], name=key, max_depth=3, max_items=20)

    else:
        # 如果顶层不是 dict，有可能是直接保存的模型/JIT对象
        print_divider("Non-dict checkpoint details")
        print("This file may be a raw torch object, model, or scripted module rather than a checkpoint dict.")
        try:
            print(ckpt)
        except Exception as e:
            print(f"Cannot print object directly: {e}")

if __name__ == "__main__":
    #checkpoint_path = r"/media/ubuntu20/D/NvidiaIsaac/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/2026-01-24_22-42-45/model_999.pt"
    checkpoint_path = r"/media/ubuntu20/D/robotic/复现/复现/result/iter_000800.pt"
    inspect_checkpoint(checkpoint_path)