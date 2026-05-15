from __future__ import annotations

import argparse
import pathlib
import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import yaml

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.utils.crc import CRC


BUTTON_BITS = {
    "R1": 0,
    "L1": 1,
    "START": 2,
    "SELECT": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "UP": 12,
    "RIGHT": 13,
    "DOWN": 14,
    "LEFT": 15,
}


class ControlPhase(str, Enum):
    WAIT_FOR_HARD_RESET = "wait_for_hard_reset"
    WAIT_FOR_START = "wait_for_start"
    BOOT_TO_DEFAULT = "boot_to_default"
    MOVE_TO_TARGET = "move_to_target"
    HOLD_TARGET = "hold_target"
    POLICY_CONTROL = "policy_control"
    RETURN_TO_DEFAULT = "return_to_default"


@dataclass
class DeployConfig:
    pose_reference_order: str
    joint_ids_map: np.ndarray
    default_joint_pos: np.ndarray
    target_joint_pos: np.ndarray
    stiffness_sdk: np.ndarray
    damping_sdk: np.ndarray
    action_clip_low: np.ndarray
    action_clip_high: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    controller_mode: str
    onnx_model_path: pathlib.Path | None
    onnx_input_name: str
    onnx_output_name: str
    onnx_startup_pose_source: str
    command_skill_id: int
    command_one_hot: np.ndarray
    fixed_velocity_command: np.ndarray
    observation_order: tuple[str, ...]
    observation_scales: dict[str, np.ndarray]
    observation_clip_low: dict[str, np.ndarray]
    observation_clip_high: dict[str, np.ndarray]
    policy_obs_dim: int


def _parse_fixed_command_value(value: object) -> float:
    """把 deploy.yaml 中的速度命令配置解析成一个固定标量。"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    array_value = np.asarray(value, dtype=np.float32).reshape(-1)
    if array_value.size == 0:
        return 0.0
    if array_value.size == 1:
        return float(array_value[0])
    if array_value.size == 2:
        return float(0.5 * (array_value[0] + array_value[1]))

    raise ValueError(f"Unsupported command specification: {value}")


def _parse_clip_vector(clip_cfg: object, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """把 clip 配置展开成逐维的 low / high 数组。"""
    if clip_cfg is None:
        low = np.full(dim, -np.inf, dtype=np.float32)
        high = np.full(dim, np.inf, dtype=np.float32)
        return low, high

    clip_array = np.asarray(clip_cfg, dtype=np.float32)
    if clip_array.ndim == 1 and clip_array.shape[0] == 2:
        low = np.full(dim, float(clip_array[0]), dtype=np.float32)
        high = np.full(dim, float(clip_array[1]), dtype=np.float32)
        return low, high
    if clip_array.ndim == 2 and clip_array.shape == (dim, 2):
        return clip_array[:, 0].astype(np.float32), clip_array[:, 1].astype(np.float32)

    raise ValueError(f"Unsupported clip specification for dim={dim}: {clip_cfg}")


def quat_wxyz_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """把 wxyz 四元数转换为 body->world 旋转矩阵。"""
    quat = np.asarray(quaternion_wxyz, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    quat = quat / norm
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def compute_projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """
    把世界系重力 `[0, 0, -1]` 投影到机体系。

    对应训练时的 `projected_gravity` 观测项。
    """
    rotation_body_to_world = quat_wxyz_to_rotation_matrix(quaternion_wxyz)
    gravity_world = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return rotation_body_to_world.T @ gravity_world


def apply_observation_postprocess(
    cfg: DeployConfig,
    term_name: str,
    raw_value: np.ndarray,
) -> np.ndarray:
    """按 deploy.yaml 中 observations.* 的 clip / scale 处理单个观测项。"""
    value = np.asarray(raw_value, dtype=np.float32).reshape(-1)
    clip_low = cfg.observation_clip_low[term_name]
    clip_high = cfg.observation_clip_high[term_name]
    scale = cfg.observation_scales[term_name]
    value = np.clip(value, clip_low, clip_high)
    return value * scale


def infer_policy_observation_dim(cfg: DeployConfig) -> int:
    return int(sum(scale.size for scale in cfg.observation_scales.values()) + cfg.command_one_hot.size)


def build_policy_observation_from_state(
    cfg: DeployConfig,
    joint_pos_policy: np.ndarray,
    joint_vel_policy: np.ndarray,
    imu_quaternion_wxyz: np.ndarray,
    imu_gyro: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """
    从部署侧可观测状态重建训练时 policy 使用的观测。

    当前约定：
    - 先拼出 45 维连续观测项
    - 再在末尾追加 skill one-hot，形成最终 49 维 policy obs
    """
    observation_terms_raw = {
        "base_ang_vel": np.asarray(imu_gyro, dtype=np.float32).reshape(-1),
        "projected_gravity": compute_projected_gravity(imu_quaternion_wxyz),
        "velocity_commands": np.asarray(cfg.fixed_velocity_command, dtype=np.float32).reshape(-1),
        "joint_pos_rel": np.asarray(joint_pos_policy, dtype=np.float32).reshape(-1) - cfg.action_offset,
        "joint_vel_rel": np.asarray(joint_vel_policy, dtype=np.float32).reshape(-1),
        "last_action": np.asarray(last_action, dtype=np.float32).reshape(-1),
    }

    observation_parts: list[np.ndarray] = []
    for term_name in cfg.observation_order:
        if term_name not in observation_terms_raw:
            raise KeyError(
                f"Unsupported observation term in deploy.yaml: '{term_name}'. "
                "Please extend deploy/go2_controller.py to rebuild this term."
            )
        observation_parts.append(
            apply_observation_postprocess(cfg, term_name, observation_terms_raw[term_name])
        )
    if cfg.command_one_hot.size > 0:
        observation_parts.append(cfg.command_one_hot.astype(np.float32, copy=False))
    return np.concatenate(observation_parts, axis=0).astype(np.float32, copy=False)


def load_deploy_config(config_path: pathlib.Path) -> DeployConfig:
    def sdk_to_policy_order(values: np.ndarray, joint_ids: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)[joint_ids]

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pose_reference_order = str(cfg.get("pose_reference_order", "policy")).strip().lower()
    if pose_reference_order not in {"policy", "sdk"}:
        raise ValueError(
            "pose_reference_order must be either 'policy' or 'sdk'. "
            f"Got '{pose_reference_order}'."
        )

    joint_ids_map = np.asarray(cfg["joint_ids_map"], dtype=np.int64).reshape(-1)
    default_joint_pos = np.asarray(cfg["default_joint_pos"], dtype=np.float32).reshape(-1)
    target_joint_pos = np.asarray(cfg["target_joint_pos"], dtype=np.float32).reshape(-1)
    stiffness_sdk = np.asarray(cfg["stiffness"], dtype=np.float32).reshape(-1)
    damping_sdk = np.asarray(cfg["damping"], dtype=np.float32).reshape(-1)

    action_cfg = cfg["actions"]["JointPositionAction"]
    action_clip_low, action_clip_high = _parse_clip_vector(
        action_cfg.get("clip", None),
        dim=joint_ids_map.shape[0],
    )
    action_scale = np.asarray(action_cfg["scale"], dtype=np.float32).reshape(-1)
    action_offset = np.asarray(action_cfg["offset"], dtype=np.float32).reshape(-1)

    expected_dim = joint_ids_map.shape[0]
    for name, value in (
        ("default_joint_pos", default_joint_pos),
        ("target_joint_pos", target_joint_pos),
        ("stiffness_sdk", stiffness_sdk),
        ("damping_sdk", damping_sdk),
        ("action_clip_low", action_clip_low),
        ("action_clip_high", action_clip_high),
        ("action_scale", action_scale),
        ("action_offset", action_offset),
    ):
        if value.shape[0] != expected_dim:
            raise ValueError(
                f"{name} dimension mismatch: expected {expected_dim}, got {value.shape[0]}"
            )

    sorted_joint_ids = np.sort(joint_ids_map)
    expected_joint_ids = np.arange(expected_dim, dtype=np.int64)
    if not np.array_equal(sorted_joint_ids, expected_joint_ids):
        raise ValueError(
            "joint_ids_map must be a permutation of [0, ..., N-1]. "
            f"Got {joint_ids_map.tolist()}"
        )

    if pose_reference_order == "sdk":
        default_joint_pos = sdk_to_policy_order(default_joint_pos, joint_ids_map)
        target_joint_pos = sdk_to_policy_order(target_joint_pos, joint_ids_map)
        action_clip_low = sdk_to_policy_order(action_clip_low, joint_ids_map)
        action_clip_high = sdk_to_policy_order(action_clip_high, joint_ids_map)
        action_scale = sdk_to_policy_order(action_scale, joint_ids_map)
        action_offset = sdk_to_policy_order(action_offset, joint_ids_map)

    controller_cfg = cfg.get("controller", {}) or {}
    controller_mode = str(controller_cfg.get("mode", "interpolation")).strip().lower()
    if controller_mode not in {"interpolation", "onnx_policy"}:
        raise ValueError(
            "controller.mode must be one of {'interpolation', 'onnx_policy'}. "
            f"Got '{controller_mode}'."
        )

    onnx_model_path_cfg = controller_cfg.get("onnx_model_path", None)
    onnx_model_path: pathlib.Path | None = None
    if onnx_model_path_cfg is not None:
        onnx_model_path = pathlib.Path(onnx_model_path_cfg).expanduser()
        if not onnx_model_path.is_absolute():
            onnx_model_path = (config_path.parent / onnx_model_path).resolve()
        else:
            onnx_model_path = onnx_model_path.resolve()

    onnx_input_name = str(controller_cfg.get("onnx_input_name", "observations"))
    onnx_output_name = str(controller_cfg.get("onnx_output_name", "actions"))
    onnx_startup_pose_source = str(controller_cfg.get("onnx_startup_pose_source", "direct")).strip().lower()
    if onnx_startup_pose_source not in {"direct", "action_offset", "target_joint_pos", "default_joint_pos"}:
        raise ValueError(
            "controller.onnx_startup_pose_source must be one of "
            "{'direct', 'action_offset', 'target_joint_pos', 'default_joint_pos'}. "
            f"Got '{onnx_startup_pose_source}'."
        )
    policy_command_cfg = cfg.get("policy_command", {}) or {}
    command_skill_id = int(policy_command_cfg.get("skill_id", 0))
    command_one_hot = np.asarray(policy_command_cfg.get("one_hot", []), dtype=np.float32).reshape(-1)

    command_cfg = cfg.get("commands", {}).get("base_velocity", {}).get("ranges", {})
    fixed_velocity_command = np.asarray(
        [
            _parse_fixed_command_value(command_cfg.get("lin_vel_x", 0.0)),
            _parse_fixed_command_value(command_cfg.get("lin_vel_y", 0.0)),
            _parse_fixed_command_value(command_cfg.get("ang_vel_z", 0.0)),
        ],
        dtype=np.float32,
    )

    observation_order: list[str] = []
    observation_scales: dict[str, np.ndarray] = {}
    observation_clip_low: dict[str, np.ndarray] = {}
    observation_clip_high: dict[str, np.ndarray] = {}
    joint_like_terms = {"joint_pos_rel", "joint_vel_rel", "last_action"}
    for term_name, term_cfg in (cfg.get("observations", {}) or {}).items():
        history_length = int(term_cfg.get("history_length", 1))
        if history_length != 1:
            raise ValueError(
                f"Only history_length=1 is currently supported for deployment. "
                f"Got observations.{term_name}.history_length={history_length}."
            )

        term_scale = np.asarray(term_cfg.get("scale", []), dtype=np.float32).reshape(-1)
        if term_scale.size == 0:
            raise ValueError(f"observations.{term_name}.scale must not be empty.")
        term_clip_low, term_clip_high = _parse_clip_vector(term_cfg.get("clip", None), dim=term_scale.size)

        if pose_reference_order == "sdk" and term_name in joint_like_terms and term_scale.size == expected_dim:
            term_scale = sdk_to_policy_order(term_scale, joint_ids_map)
            term_clip_low = sdk_to_policy_order(term_clip_low, joint_ids_map)
            term_clip_high = sdk_to_policy_order(term_clip_high, joint_ids_map)

        observation_order.append(term_name)
        observation_scales[term_name] = term_scale
        observation_clip_low[term_name] = term_clip_low
        observation_clip_high[term_name] = term_clip_high

    policy_obs_dim = int(sum(scale.size for scale in observation_scales.values()) + command_one_hot.size)
    if controller_mode == "onnx_policy" and command_one_hot.size == 0:
        raise ValueError(
            "controller.mode='onnx_policy' requires deploy.yaml policy_command.one_hot, "
            "because the trained policy observation appends skill one-hot at the end."
        )

    return DeployConfig(
        pose_reference_order=pose_reference_order,
        joint_ids_map=joint_ids_map,
        default_joint_pos=default_joint_pos,
        target_joint_pos=target_joint_pos,
        stiffness_sdk=stiffness_sdk,
        damping_sdk=damping_sdk,
        action_clip_low=action_clip_low,
        action_clip_high=action_clip_high,
        action_scale=action_scale,
        action_offset=action_offset,
        controller_mode=controller_mode,
        onnx_model_path=onnx_model_path,
        onnx_input_name=onnx_input_name,
        onnx_output_name=onnx_output_name,
        onnx_startup_pose_source=onnx_startup_pose_source,
        command_skill_id=command_skill_id,
        command_one_hot=command_one_hot,
        fixed_velocity_command=fixed_velocity_command,
        observation_order=tuple(observation_order),
        observation_scales=observation_scales,
        observation_clip_low=observation_clip_low,
        observation_clip_high=observation_clip_high,
        policy_obs_dim=policy_obs_dim,
    )


class Go2Controller:
    """
    First-stage deployment controller for debugging the DDS and joint mapping chain.

    Behavior:
    1. Wait for the first lowstate message.
    2. Optionally wait until MuJoCo performs a simulator-side hard reset.
    3. Hold default_joint_pos after the hard reset.
    4. Press the start-follow button to move from default_joint_pos to target_joint_pos.
    5. Once the measured error is small enough, optionally latch the actually reached
       pose and hold it with softer gains instead of continuing to push the nominal
       target_joint_pos aggressively.
    6. Press the cycle reset button to return to default_joint_pos and wait for the next start-follow button.
    7. Optionally listen for a dedicated hard reset button, so the controller can
       re-synchronize itself when MuJoCo has already done a simulator-side hard reset.

    Controller modes:
    - interpolation: use deploy.yaml target pose and move slowly by joint interpolation
    - onnx_policy: rebuild the trained policy observation online and infer actions from policy.onnx
    """

    def __init__(
        self,
        config_path: pathlib.Path,
        domain_id: int = 0,
        network: str = "lo",
        control_dt: float = 0.05,
        max_joint_step: float = 0.005,
        reach_tolerance: float = 0.05,
        reset_button: str = "A",
        hard_reset_button: str | None = None,
        start_follow_button: str = "X",
        wait_for_initial_hard_reset: bool = True,
        print_every: float = 1.0,
        pose_kp_hip: float = 15.0,
        pose_kp_thigh: float = 25.0,
        pose_kp_calf: float = 25.0,
        pose_kd_hip: float = 1.5,
        pose_kd_thigh: float = 2.0,
        pose_kd_calf: float = 2.0,
        hold_kp_scale: float = 0.35,
        hold_kd_scale: float = 0.5,
        latch_reached_pose_on_target: bool = True,
        snap_to_default_on_start: bool = True,
    ) -> None:
        self.config_path = config_path.resolve()
        self.cfg = load_deploy_config(self.config_path)

        self.domain_id = int(domain_id)
        self.network = str(network)
        self.control_dt = float(control_dt)
        self.max_joint_step = float(max_joint_step)
        self.reach_tolerance = float(reach_tolerance)
        self.reset_button = reset_button.upper()
        self.hard_reset_button = (
            hard_reset_button.upper() if hard_reset_button is not None else None
        )
        self.start_follow_button = start_follow_button.upper()
        self.wait_for_initial_hard_reset = bool(wait_for_initial_hard_reset)
        self.print_every = float(print_every)
        self.snap_to_default_on_start = bool(snap_to_default_on_start)
        self.hold_kp_scale = float(hold_kp_scale)
        self.hold_kd_scale = float(hold_kd_scale)
        self.latch_reached_pose_on_target = bool(latch_reached_pose_on_target)
        self.controller_mode = self.cfg.controller_mode
        self.pose_kp_sdk = np.asarray(
            [
                pose_kp_hip, pose_kp_thigh, pose_kp_calf,
                pose_kp_hip, pose_kp_thigh, pose_kp_calf,
                pose_kp_hip, pose_kp_thigh, pose_kp_calf,
                pose_kp_hip, pose_kp_thigh, pose_kp_calf,
            ],
            dtype=np.float32,
        )
        self.pose_kd_sdk = np.asarray(
            [
                pose_kd_hip, pose_kd_thigh, pose_kd_calf,
                pose_kd_hip, pose_kd_thigh, pose_kd_calf,
                pose_kd_hip, pose_kd_thigh, pose_kd_calf,
                pose_kd_hip, pose_kd_thigh, pose_kd_calf,
            ],
            dtype=np.float32,
        )
        self.hold_pose_kp_sdk = self.pose_kp_sdk * self.hold_kp_scale
        self.hold_pose_kd_sdk = self.pose_kd_sdk * self.hold_kd_scale

        if self.reset_button not in BUTTON_BITS:
            raise ValueError(
                f"Unsupported reset button '{reset_button}'. "
                f"Supported buttons: {sorted(BUTTON_BITS.keys())}"
            )
        if self.start_follow_button not in BUTTON_BITS:
            raise ValueError(
                f"Unsupported start follow button '{start_follow_button}'. "
                f"Supported buttons: {sorted(BUTTON_BITS.keys())}"
            )
        if self.hard_reset_button is not None and self.hard_reset_button not in BUTTON_BITS:
            raise ValueError(
                f"Unsupported hard reset button '{hard_reset_button}'. "
                f"Supported buttons: {sorted(BUTTON_BITS.keys())}"
            )
        if self.hard_reset_button is not None and self.hard_reset_button == self.reset_button:
            raise ValueError(
                "hard_reset_button must be different from reset_button, "
                "otherwise the controller-side soft reset and simulator-side hard reset "
                "will be triggered by the same key."
            )
        if self.start_follow_button == self.reset_button:
            raise ValueError("start_follow_button must be different from reset_button.")
        if self.hard_reset_button is not None and self.start_follow_button == self.hard_reset_button:
            raise ValueError("start_follow_button must be different from hard_reset_button.")
        if self.wait_for_initial_hard_reset and self.hard_reset_button is None:
            raise ValueError(
                "wait_for_initial_hard_reset=True requires hard_reset_button to be set."
            )

        self._lock = threading.Lock()
        self._joint_pos_sdk: np.ndarray | None = None
        self._joint_vel_sdk: np.ndarray | None = None
        self._imu_quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._imu_gyro = np.zeros(3, dtype=np.float32)
        self._wireless_keys: int = 0

        self._prev_button_states: dict[str, bool] = {}
        for button_name in (self.reset_button, self.hard_reset_button, self.start_follow_button):
            if button_name is not None:
                self._prev_button_states[button_name] = False
        self._running = True
        self._lowstate_count = 0
        self._publish_count = 0
        self._first_lowstate_reported = False

        self.phase = ControlPhase.BOOT_TO_DEFAULT
        self.commanded_pose_policy: np.ndarray | None = None
        self._last_policy_action = np.zeros_like(self.cfg.default_joint_pos, dtype=np.float32)
        self._follow_target_pose_policy: np.ndarray | None = None
        self._enter_policy_after_target = False
        self._onnx_session = None
        self._onnx_input_name = self.cfg.onnx_input_name
        self._onnx_output_name = self.cfg.onnx_output_name

        self._crc = CRC()
        self._low_cmd = unitree_go_msg_dds__LowCmd_()
        self._init_low_cmd_template()

        if self.controller_mode == "onnx_policy":
            self._init_onnx_policy()

        self._init_channels()

    def _init_channels(self) -> None:
        if self.network:
            ChannelFactoryInitialize(self.domain_id, self.network)
        else:
            ChannelFactoryInitialize(self.domain_id)

        self._lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._lowcmd_publisher.Init()

        self._lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_subscriber.Init(self._lowstate_handler, 10)

        self._wireless_subscriber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self._wireless_subscriber.Init(self._wireless_handler, 10)

    def _init_low_cmd_template(self) -> None:
        self._low_cmd.head[0] = 0xFE
        self._low_cmd.head[1] = 0xEF
        self._low_cmd.level_flag = 0xFF
        self._low_cmd.gpio = 0

        for i in range(len(self._low_cmd.motor_cmd)):
            motor = self._low_cmd.motor_cmd[i]
            motor.mode = 0x01
            motor.q = 0.0
            motor.kp = 0.0
            motor.dq = 0.0
            motor.kd = 0.0
            motor.tau = 0.0

        # Initialize the low-level command with the nominal pose gains. During HOLD_TARGET
        # we may switch to softer gains to reduce chatter around the reached pose.
        for sdk_id in range(min(len(self.pose_kp_sdk), len(self._low_cmd.motor_cmd))):
            motor = self._low_cmd.motor_cmd[sdk_id]
            motor.kp = float(self.pose_kp_sdk[sdk_id])
            motor.kd = float(self.pose_kd_sdk[sdk_id])

    def _lowstate_handler(self, msg: LowState_) -> None:
        joint_count = len(msg.motor_state)
        joint_pos = np.zeros(joint_count, dtype=np.float32)
        joint_vel = np.zeros(joint_count, dtype=np.float32)

        for i in range(joint_count):
            joint_pos[i] = float(msg.motor_state[i].q)
            joint_vel[i] = float(msg.motor_state[i].dq)

        imu_quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        imu_gyro = np.zeros(3, dtype=np.float32)
        imu_state = getattr(msg, "imu_state", None)
        if imu_state is not None:
            try:
                imu_quaternion_wxyz = np.asarray(imu_state.quaternion[:4], dtype=np.float32).reshape(4)
            except Exception:
                imu_quaternion_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            try:
                imu_gyro = np.asarray(imu_state.gyroscope[:3], dtype=np.float32).reshape(3)
            except Exception:
                imu_gyro = np.zeros(3, dtype=np.float32)

        with self._lock:
            self._joint_pos_sdk = joint_pos
            self._joint_vel_sdk = joint_vel
            self._imu_quaternion_wxyz = imu_quaternion_wxyz
            self._imu_gyro = imu_gyro
            self._lowstate_count += 1

            if not self._first_lowstate_reported:
                first_joint = float(joint_pos[0]) if joint_pos.size > 0 else 0.0
                print(
                    "[INFO] First rt/lowstate received. "
                    f"joint_count={joint_count}, first_joint_q={first_joint:.4f}"
                )
                self._first_lowstate_reported = True

    def _wireless_handler(self, msg: WirelessController_) -> None:
        with self._lock:
            self._wireless_keys = int(msg.keys)

    def wait_for_lowstate(self, timeout_s: float | None = None) -> None:
        start_time = time.perf_counter()
        while self._running:
            with self._lock:
                ready = self._joint_pos_sdk is not None
            if ready:
                return
            if timeout_s is not None and (time.perf_counter() - start_time) > timeout_s:
                raise TimeoutError(
                    "Timed out while waiting for rt/lowstate. "
                    "Please confirm that simulate_python is running, DOMAIN_ID matches, "
                    "and both sides use the same network interface, for example '--network lo'."
                )
            time.sleep(0.05)

    def stop(self) -> None:
        self._running = False

    def _init_onnx_policy(self) -> None:
        """按 deploy.yaml 配置初始化 ONNX 推理会话。"""
        if self.cfg.onnx_model_path is None:
            raise ValueError(
                "controller.mode='onnx_policy' requires controller.onnx_model_path in deploy.yaml."
            )
        if not self.cfg.onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.cfg.onnx_model_path}")

        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "controller.mode='onnx_policy' requires the 'onnxruntime' package. "
                "Please install it in the unitree_rl environment first."
            ) from exc

        self._onnx_session = ort.InferenceSession(
            str(self.cfg.onnx_model_path),
            providers=["CPUExecutionProvider"],
        )
        available_inputs = [item.name for item in self._onnx_session.get_inputs()]
        available_outputs = [item.name for item in self._onnx_session.get_outputs()]
        if self._onnx_input_name not in available_inputs and available_inputs:
            self._onnx_input_name = available_inputs[0]
        if self._onnx_output_name not in available_outputs and available_outputs:
            self._onnx_output_name = available_outputs[0]

        input_shape = self._onnx_session.get_inputs()[0].shape
        if input_shape and isinstance(input_shape[-1], int):
            model_obs_dim = int(input_shape[-1])
            if model_obs_dim != int(self.cfg.policy_obs_dim):
                raise ValueError(
                    "ONNX input dim 与 deploy 重建的 policy obs dim 不一致："
                    f" model={model_obs_dim}, deploy={self.cfg.policy_obs_dim}."
                    " 请检查 deploy.yaml 的 observations.* 和 policy_command.one_hot 是否与训练时一致。"
                )

    def _sdk_to_policy_order(self, sdk_values: np.ndarray) -> np.ndarray:
        return np.asarray(sdk_values, dtype=np.float32)[self.cfg.joint_ids_map]

    def _policy_to_sdk_order(self, policy_values: np.ndarray) -> np.ndarray:
        sdk_values = np.zeros(len(self._low_cmd.motor_cmd), dtype=np.float32)
        sdk_values[self.cfg.joint_ids_map] = np.asarray(policy_values, dtype=np.float32)
        return sdk_values

    def _get_joint_pos_policy(self) -> np.ndarray:
        with self._lock:
            if self._joint_pos_sdk is None:
                raise RuntimeError("Lowstate has not been received yet.")
            joint_pos_sdk = self._joint_pos_sdk.copy()
        return self._sdk_to_policy_order(joint_pos_sdk)

    def _get_joint_vel_policy(self) -> np.ndarray:
        with self._lock:
            if self._joint_vel_sdk is None:
                raise RuntimeError("Lowstate has not been received yet.")
            joint_vel_sdk = self._joint_vel_sdk.copy()
        return self._sdk_to_policy_order(joint_vel_sdk)

    def _get_imu_state(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            imu_quaternion = self._imu_quaternion_wxyz.copy()
            imu_gyro = self._imu_gyro.copy()
        return imu_quaternion, imu_gyro

    def _consume_button_edge(self, button_name: str | None) -> bool:
        if button_name is None:
            return False

        mask = 1 << BUTTON_BITS[button_name]
        with self._lock:
            pressed = bool(self._wireless_keys & mask)

        previously_pressed = self._prev_button_states.get(button_name, False)
        edge = pressed and not previously_pressed
        self._prev_button_states[button_name] = pressed
        return edge

    @staticmethod
    def _quat_wxyz_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
        return quat_wxyz_to_rotation_matrix(quaternion_wxyz)

    def _compute_projected_gravity(self, quaternion_wxyz: np.ndarray) -> np.ndarray:
        return compute_projected_gravity(quaternion_wxyz)

    def _apply_observation_postprocess(self, term_name: str, raw_value: np.ndarray) -> np.ndarray:
        return apply_observation_postprocess(self.cfg, term_name, raw_value)

    def _build_policy_observation(self) -> np.ndarray:
        """从实时 lowstate 重建训练时 policy 使用的 49 维观测。"""
        joint_pos_policy = self._get_joint_pos_policy()
        joint_vel_policy = self._get_joint_vel_policy()
        imu_quaternion, imu_gyro = self._get_imu_state()
        return build_policy_observation_from_state(
            cfg=self.cfg,
            joint_pos_policy=joint_pos_policy,
            joint_vel_policy=joint_vel_policy,
            imu_quaternion_wxyz=imu_quaternion,
            imu_gyro=imu_gyro,
            last_action=self._last_policy_action,
        )

    def _action_to_target_pose(self, action_policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        把 ONNX actor 输出的策略动作转为部署目标关节位置。

        返回：
        - `clipped_action`: 裁剪后的原始策略动作
        - `target_pose_policy`: 通过 offset + scale * action 得到的目标关节位置
        """
        action = np.asarray(action_policy, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.cfg.default_joint_pos.shape[0]:
            raise ValueError(
                f"Policy action dim mismatch: expected {self.cfg.default_joint_pos.shape[0]}, got {action.shape[0]}"
            )
        clipped_action = np.clip(action, self.cfg.action_clip_low, self.cfg.action_clip_high)
        target_pose_policy = self.cfg.action_offset + self.cfg.action_scale * clipped_action
        return clipped_action.astype(np.float32), target_pose_policy.astype(np.float32)

    def _infer_policy_target_pose(self) -> np.ndarray:
        """执行一次 ONNX 推理，并返回本周期期望的目标关节姿态。"""
        if self._onnx_session is None:
            raise RuntimeError("ONNX session has not been initialized.")

        observation = self._build_policy_observation()
        onnx_outputs = self._onnx_session.run(
            [self._onnx_output_name],
            {self._onnx_input_name: observation.reshape(1, -1).astype(np.float32)},
        )
        clipped_action, target_pose_policy = self._action_to_target_pose(onnx_outputs[0][0])
        self._last_policy_action = clipped_action
        return target_pose_policy

    def _goal_for_phase(self) -> np.ndarray | None:
        if self.phase == ControlPhase.WAIT_FOR_HARD_RESET:
            return None
        if self.phase in (
            ControlPhase.WAIT_FOR_START,
            ControlPhase.BOOT_TO_DEFAULT,
            ControlPhase.RETURN_TO_DEFAULT,
        ):
            return self.cfg.default_joint_pos
        if self.phase == ControlPhase.POLICY_CONTROL and self.commanded_pose_policy is not None:
            return self.commanded_pose_policy
        if self.phase in (ControlPhase.MOVE_TO_TARGET, ControlPhase.HOLD_TARGET):
            if self._follow_target_pose_policy is not None:
                return self._follow_target_pose_policy
        if self.phase == ControlPhase.HOLD_TARGET and self.commanded_pose_policy is not None:
            return self.commanded_pose_policy
        return self.cfg.target_joint_pos

    def _resolve_onnx_startup_pose_policy(self) -> np.ndarray | None:
        source = self.cfg.onnx_startup_pose_source
        if source == "direct":
            return None
        if source == "action_offset":
            return self.cfg.action_offset.copy()
        if source == "target_joint_pos":
            return self.cfg.target_joint_pos.copy()
        if source == "default_joint_pos":
            return self.cfg.default_joint_pos.copy()
        raise ValueError(f"Unsupported onnx_startup_pose_source: {source}")

    def _begin_follow_sequence(self) -> None:
        self._last_policy_action[:] = 0.0
        if self.controller_mode != "onnx_policy":
            self._follow_target_pose_policy = self.cfg.target_joint_pos.copy()
            self._enter_policy_after_target = False
            self.phase = ControlPhase.MOVE_TO_TARGET
            print("[INFO] Start-follow button pressed. Moving toward target_joint_pos.")
            return

        startup_pose_policy = self._resolve_onnx_startup_pose_policy()
        if startup_pose_policy is None:
            self._follow_target_pose_policy = None
            self._enter_policy_after_target = False
            self.phase = ControlPhase.POLICY_CONTROL
            print("[INFO] Start-follow button pressed. Switching to ONNX policy control.")
            return

        self._follow_target_pose_policy = startup_pose_policy
        self._enter_policy_after_target = True
        self.phase = ControlPhase.MOVE_TO_TARGET
        print(
            "[INFO] Start-follow button pressed. "
            f"Moving toward ONNX startup pose source='{self.cfg.onnx_startup_pose_source}' before policy control."
        )

    def _step_command_towards_goal(self, goal_policy: np.ndarray) -> np.ndarray:
        if self.commanded_pose_policy is None:
            self.commanded_pose_policy = goal_policy.copy()
            return self.commanded_pose_policy.copy()

        delta = goal_policy - self.commanded_pose_policy
        delta = np.clip(delta, -self.max_joint_step, self.max_joint_step)
        self.commanded_pose_policy = self.commanded_pose_policy + delta
        return self.commanded_pose_policy.copy()

    def _publish_policy_pose(self, commanded_pose_policy: np.ndarray) -> None:
        commanded_pose_sdk = self._policy_to_sdk_order(commanded_pose_policy)
        kp_values = self.pose_kp_sdk
        kd_values = self.pose_kd_sdk
        if self.phase == ControlPhase.HOLD_TARGET:
            kp_values = self.hold_pose_kp_sdk
            kd_values = self.hold_pose_kd_sdk

        # This controller currently drives only the leg joints described by deploy.yaml.
        # Some SDK message definitions expose additional motor slots, so we must avoid
        # indexing the 12-dim pose gains with those extra entries.
        controlled_joint_count = min(
            len(self.cfg.default_joint_pos),
            len(commanded_pose_sdk),
            len(self._low_cmd.motor_cmd),
            len(kp_values),
            len(kd_values),
        )

        for sdk_id in range(controlled_joint_count):
            motor = self._low_cmd.motor_cmd[sdk_id]
            motor.q = float(commanded_pose_sdk[sdk_id])
            motor.kp = float(kp_values[sdk_id])
            motor.dq = 0.0
            motor.kd = float(kd_values[sdk_id])
            motor.tau = 0.0

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._lowcmd_publisher.Write(self._low_cmd)
        self._publish_count += 1

    def _maybe_transition_phase(self, current_pose_policy: np.ndarray) -> None:
        goal_policy = self._goal_for_phase()
        if goal_policy is None:
            return
        measured_error = float(np.max(np.abs(current_pose_policy - goal_policy)))
        command_error = (
            float(np.max(np.abs(self.commanded_pose_policy - goal_policy)))
            if self.commanded_pose_policy is not None
            else float("inf")
        )

        if self.phase == ControlPhase.BOOT_TO_DEFAULT:
            if measured_error <= self.reach_tolerance and command_error <= self.reach_tolerance:
                self.phase = ControlPhase.WAIT_FOR_START
                self.commanded_pose_policy = self.cfg.default_joint_pos.copy()
                print(
                    "[INFO] Reached default_joint_pos. "
                    f"Press {self.start_follow_button} to begin following target_joint_pos."
                )
            return

        if self.phase == ControlPhase.MOVE_TO_TARGET:
            if measured_error <= self.reach_tolerance and command_error <= self.reach_tolerance:
                if self._enter_policy_after_target:
                    self.phase = ControlPhase.POLICY_CONTROL
                    self.commanded_pose_policy = current_pose_policy.copy()
                    self._enter_policy_after_target = False
                    print(
                        "[INFO] Reached ONNX startup pose tolerance. "
                        "Switching to ONNX policy control."
                    )
                    return

                self.phase = ControlPhase.HOLD_TARGET
                if self.latch_reached_pose_on_target:
                    # Lock to the pose that was actually reached, which is usually more
                    # stable than continuing to push the exact nominal target forever.
                    self.commanded_pose_policy = current_pose_policy.copy()
                    print(
                        "[INFO] Reached target_joint_pos tolerance. "
                        "Latching the reached pose and switching to softer hold gains."
                    )
                else:
                    self.commanded_pose_policy = self.cfg.target_joint_pos.copy()
                    print(
                        "[INFO] Reached target_joint_pos. "
                        "Holding nominal target pose and waiting for reset button."
                    )
            return

        if self.phase == ControlPhase.RETURN_TO_DEFAULT:
            if measured_error <= self.reach_tolerance and command_error <= self.reach_tolerance:
                self.phase = ControlPhase.WAIT_FOR_START
                self.commanded_pose_policy = self.cfg.default_joint_pos.copy()
                self._follow_target_pose_policy = None
                self._enter_policy_after_target = False
                print(
                    "[INFO] Reset complete. Holding default_joint_pos. "
                    f"Press {self.start_follow_button} to start the next follow cycle."
                )

    def run(self) -> None:
        print(f"[INFO] Loading config: {self.config_path}")
        print(f"[INFO] controller_mode={self.controller_mode}")
        print(f"[INFO] DDS domain_id={self.domain_id}, network='{self.network}'")
        print(f"[INFO] control_dt={self.control_dt:.3f}s, max_joint_step={self.max_joint_step:.4f} rad")
        print(
            f"[INFO] reach_tolerance={self.reach_tolerance:.4f} rad, "
            f"cycle_reset_button={self.reset_button}, "
            f"hard_reset_button={self.hard_reset_button}, "
            f"start_follow_button={self.start_follow_button}"
        )
        print(f"[INFO] pose_reference_order={self.cfg.pose_reference_order}")
        print(f"[INFO] wait_for_initial_hard_reset={self.wait_for_initial_hard_reset}")
        print(f"[INFO] snap_to_default_on_start={self.snap_to_default_on_start}")
        print(f"[INFO] pose_kp_sdk={np.array2string(self.pose_kp_sdk, precision=1, separator=', ')}")
        print(f"[INFO] pose_kd_sdk={np.array2string(self.pose_kd_sdk, precision=1, separator=', ')}")
        print(f"[INFO] hold_pose_kp_sdk={np.array2string(self.hold_pose_kp_sdk, precision=2, separator=', ')}")
        print(f"[INFO] hold_pose_kd_sdk={np.array2string(self.hold_pose_kd_sdk, precision=2, separator=', ')}")
        print(f"[INFO] latch_reached_pose_on_target={self.latch_reached_pose_on_target}")
        print(f"[INFO] default_joint_pos={np.array2string(self.cfg.default_joint_pos, precision=3, separator=', ')}")
        print(f"[INFO] target_joint_pos={np.array2string(self.cfg.target_joint_pos, precision=3, separator=', ')}")
        print(
            "[INFO] fixed_velocity_command="
            f"{np.array2string(self.cfg.fixed_velocity_command, precision=3, separator=', ')}"
        )
        if self.controller_mode == "onnx_policy":
            print(f"[INFO] onnx_model_path={self.cfg.onnx_model_path}")
            print(f"[INFO] onnx_startup_pose_source={self.cfg.onnx_startup_pose_source}")
            print(
                f"[INFO] onnx_io_names=input:{self._onnx_input_name}, "
                f"output:{self._onnx_output_name}"
            )
        if self.hard_reset_button is None and self.reset_button == "A":
            print(
                "[WARN] If you are running unitree_mujoco_reset_version.py, "
                "its default simulator-side hard reset button is also 'A'. "
                "Recommended split: '--reset-button B --hard-reset-button A'."
            )
        print("[INFO] Waiting for rt/lowstate ...")

        self.wait_for_lowstate(timeout_s=5.0)
        current_pose_policy = self._get_joint_pos_policy()
        if self.wait_for_initial_hard_reset:
            startup_default_error = float(
                np.max(np.abs(current_pose_policy - self.cfg.default_joint_pos))
            )
            if startup_default_error <= self.reach_tolerance:
                self.phase = ControlPhase.WAIT_FOR_START
                self.commanded_pose_policy = self.cfg.default_joint_pos.copy()
                print(
                    "[INFO] Connected. Current pose is already close to default_joint_pos, "
                    "so the controller assumes MuJoCo was already hard-reset. "
                    f"Press {self.start_follow_button} to start following target_joint_pos."
                )
            else:
                self.phase = ControlPhase.WAIT_FOR_HARD_RESET
                self.commanded_pose_policy = None
                print(
                    "[INFO] Connected. Waiting for simulator-side hard reset first. "
                    f"Press {self.hard_reset_button} in MuJoCo reset version, then press "
                    f"{self.start_follow_button} to start following target_joint_pos."
                )
        elif self.snap_to_default_on_start:
            self.phase = ControlPhase.BOOT_TO_DEFAULT
            self.commanded_pose_policy = self.cfg.default_joint_pos.copy()
            print("[INFO] Startup mode: immediately commanding default_joint_pos.")
        else:
            self.phase = ControlPhase.BOOT_TO_DEFAULT
            self.commanded_pose_policy = current_pose_policy.copy()
            print("[INFO] Startup mode: interpolating from current pose to default_joint_pos.")

        last_print_time = 0.0

        while self._running:
            loop_start = time.perf_counter()
            current_pose_policy = self._get_joint_pos_policy()

            if self._consume_button_edge(self.hard_reset_button):
                self.phase = ControlPhase.WAIT_FOR_START
                self.commanded_pose_policy = self.cfg.default_joint_pos.copy()
                self._last_policy_action[:] = 0.0
                print(
                    "[INFO] Hard reset button observed from rt/wirelesscontroller. "
                    "Assuming MuJoCo already reset to default_joint_pos. "
                    f"Holding default pose now. Press {self.start_follow_button} to begin following target_joint_pos."
                )
            elif self._consume_button_edge(self.start_follow_button):
                if self.phase == ControlPhase.WAIT_FOR_START:
                    self._begin_follow_sequence()
            elif self._consume_button_edge(self.reset_button):
                if self.phase in (
                    ControlPhase.MOVE_TO_TARGET,
                    ControlPhase.HOLD_TARGET,
                    ControlPhase.POLICY_CONTROL,
                ):
                    self.phase = ControlPhase.RETURN_TO_DEFAULT
                    self._last_policy_action[:] = 0.0
                    self._follow_target_pose_policy = None
                    self._enter_policy_after_target = False
                    print("[INFO] Cycle reset button pressed. Returning to default_joint_pos.")

            goal_policy = self._goal_for_phase()
            if self.phase == ControlPhase.POLICY_CONTROL:
                target_pose_policy = self._infer_policy_target_pose()
                commanded_pose_policy = self._step_command_towards_goal(target_pose_policy)
                self._publish_policy_pose(commanded_pose_policy)
                goal_policy = commanded_pose_policy.copy()
            elif goal_policy is not None:
                if self.phase == ControlPhase.WAIT_FOR_START:
                    self.commanded_pose_policy = goal_policy.copy()
                elif self.phase == ControlPhase.HOLD_TARGET:
                    if self.commanded_pose_policy is None:
                        self.commanded_pose_policy = goal_policy.copy()
                elif self.phase == ControlPhase.BOOT_TO_DEFAULT and self.snap_to_default_on_start:
                    self.commanded_pose_policy = goal_policy.copy()
                else:
                    self._step_command_towards_goal(goal_policy)

                self._publish_policy_pose(self.commanded_pose_policy)
                self._maybe_transition_phase(current_pose_policy)

            now = time.perf_counter()
            if now - last_print_time >= self.print_every:
                if goal_policy is None:
                    print(
                        "[INFO] "
                        f"phase={self.phase.value} "
                        f"lowstate_count={self._lowstate_count} "
                        f"publish_count={self._publish_count} "
                        "controller_output=idle_waiting_for_hard_reset "
                        f"measured_first3={np.array2string(current_pose_policy[:3], precision=3, separator=', ')}"
                    )
                else:
                    abs_error = np.abs(current_pose_policy - goal_policy)
                    measured_error = float(np.max(abs_error))
                    worst_joint_idx = int(np.argmax(abs_error))
                    print(
                        "[INFO] "
                        f"phase={self.phase.value} "
                        f"lowstate_count={self._lowstate_count} "
                        f"publish_count={self._publish_count} "
                        f"measured_max_error={measured_error:.4f} "
                        f"worst_joint={worst_joint_idx} "
                        f"measured={current_pose_policy[worst_joint_idx]:.3f} "
                        f"goal={goal_policy[worst_joint_idx]:.3f} "
                        f"commanded_first3={np.array2string(self.commanded_pose_policy[:3], precision=3, separator=', ')} "
                        f"measured_first3={np.array2string(current_pose_policy[:3], precision=3, separator=', ')}"
                    )
                last_print_time = now

            remaining = self.control_dt - (time.perf_counter() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Go2 first-stage deployment controller. "
            "This version plays a slow pose cycle from default_joint_pos to target_joint_pos."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(pathlib.Path(__file__).with_name("deploy.yaml")),
        help="Path to deploy.yaml.",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="DDS domain id.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="lo",
        help="DDS network interface. Default is 'lo' to match simulate_python/config.py.",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.02,
        help="Controller period in seconds. Larger values make the motion easier to observe.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.002,
        help="Maximum commanded joint change per control step in radians.",
    )
    parser.add_argument(
        "--reach-tolerance",
        type=float,
        default=0.05,
        help="Pose reached threshold in radians based on max absolute joint error.",
    )
    parser.add_argument(
        "--reset-button",
        type=str,
        default="B",
        help=f"Joystick button used for the controller-side cycle reset. Supported: {sorted(BUTTON_BITS.keys())}",
    )
    parser.add_argument(
        "--hard-reset-button",
        type=str,
        default="A",
        help=(
            "Optional joystick button that means MuJoCo has already done a simulator-side hard reset. "
            "When this edge is observed, the controller immediately re-synchronizes to default_joint_pos."
        ),
    )
    parser.add_argument(
        "--start-follow-button",
        type=str,
        default="X",
        help="Joystick button used to start following target_joint_pos after the robot is back at default_joint_pos.",
    )
    parser.add_argument(
        "--wait-for-initial-hard-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, the controller stays idle until it sees the simulator-side hard reset button edge.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Print status once every N seconds.",
    )
    parser.add_argument(
        "--snap-to-default-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, command default_joint_pos immediately after the first lowstate is received.",
    )
    parser.add_argument(
        "--pose-kp-hip",
        type=float,
        default=15.0,
        help="Pose control gain for hip joints in this first-stage debug controller.",
    )
    parser.add_argument(
        "--pose-kp-thigh",
        type=float,
        default=25.0,
        help="Pose control gain for thigh joints in this first-stage debug controller.",
    )
    parser.add_argument(
        "--pose-kp-calf",
        type=float,
        default=25.0,
        help="Pose control gain for calf joints in this first-stage debug controller.",
    )
    parser.add_argument(
        "--pose-kd-hip",
        type=float,
        default=1.5,
        help="Pose damping gain for hip joints in this first-stage debug controller.",
    )
    parser.add_argument(
        "--pose-kd-thigh",
        type=float,
        default=2.0,
        help="Pose damping gain for thigh joints in this first-stage debug controller.",
    )
    parser.add_argument(
        "--pose-kd-calf",
        type=float,
        default=2.0,
        help="Pose damping gain for calf joints in this first-stage debug controller.",
    )
    parser.add_argument(
        "--hold-kp-scale",
        type=float,
        default=0.35,
        help="Scale factor applied to pose kp after reaching target pose.",
    )
    parser.add_argument(
        "--hold-kd-scale",
        type=float,
        default=0.5,
        help="Scale factor applied to pose kd after reaching target pose.",
    )
    parser.add_argument(
        "--latch-reached-pose-on-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, once target tolerance is reached the controller holds the "
            "actually reached pose instead of continuing to push the exact nominal target."
        ),
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    controller = Go2Controller(
        config_path=pathlib.Path(args.config),
        domain_id=args.domain_id,
        network=args.network,
        control_dt=args.control_dt,
        max_joint_step=args.max_joint_step,
        reach_tolerance=args.reach_tolerance,
        reset_button=args.reset_button,
        hard_reset_button=args.hard_reset_button,
        start_follow_button=args.start_follow_button,
        wait_for_initial_hard_reset=args.wait_for_initial_hard_reset,
        print_every=args.print_every,
        pose_kp_hip=args.pose_kp_hip,
        pose_kp_thigh=args.pose_kp_thigh,
        pose_kp_calf=args.pose_kp_calf,
        pose_kd_hip=args.pose_kd_hip,
        pose_kd_thigh=args.pose_kd_thigh,
        pose_kd_calf=args.pose_kd_calf,
        hold_kp_scale=args.hold_kp_scale,
        hold_kd_scale=args.hold_kd_scale,
        latch_reached_pose_on_target=args.latch_reached_pose_on_target,
        snap_to_default_on_start=args.snap_to_default_on_start,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Stopping controller.")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
