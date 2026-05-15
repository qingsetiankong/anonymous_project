from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 onnxruntime 对一批 observation 做推理。")
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型路径。")
    parser.add_argument("--input", type=str, required=True, help="输入 observation .npy 路径。")
    parser.add_argument("--output", type=str, required=True, help="输出 action .npy 路径。")
    parser.add_argument("--input-name", type=str, default="observations")
    parser.add_argument("--output-name", type=str, default="actions")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "deploy/onnx_batch_infer.py 需要 onnxruntime。"
        ) from exc

    model_path = pathlib.Path(args.model).expanduser().resolve()
    input_path = pathlib.Path(args.input).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()

    observations = np.load(input_path).astype(np.float32)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    available_inputs = [item.name for item in session.get_inputs()]
    available_outputs = [item.name for item in session.get_outputs()]
    input_name = args.input_name if args.input_name in available_inputs else available_inputs[0]
    output_name = args.output_name if args.output_name in available_outputs else available_outputs[0]

    actions = session.run([output_name], {input_name: observations})[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.asarray(actions, dtype=np.float32))


if __name__ == "__main__":
    main()
