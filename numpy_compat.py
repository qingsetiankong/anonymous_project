from __future__ import annotations


def ensure_numpy_legacy_aliases() -> None:
    """
    为依赖老 NumPy API 的 Isaac Gym / legged_gym 代码补齐移除的别名。

    NumPy 1.24 移除了 `np.float`、`np.int` 等历史别名，但 Isaac Gym 及其
    周边老代码仍然会直接引用这些名字。这里在进程启动早期补一个轻量兼容层，
    避免修改外部安装目录。
    """
    import numpy as np

    legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
        "long": int,
    }

    for alias_name, alias_target in legacy_aliases.items():
        if alias_name not in np.__dict__:
            setattr(np, alias_name, alias_target)
