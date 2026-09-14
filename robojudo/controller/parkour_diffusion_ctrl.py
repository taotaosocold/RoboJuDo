from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from robojudo.controller import ControllerHook, ctrl_registry
from robojudo.controller.parkour_diffusion import FlowMatchingSampler
from robojudo.utils.util_func import quatToEuler

logger = logging.getLogger(__name__)

DIFFUSION_JOINT_NAMES = (
    "left_leg_pelvic_pitch_joint",
    "right_leg_pelvic_pitch_joint",
    "waist_yaw_joint",
    "left_leg_pelvic_roll_joint",
    "right_leg_pelvic_roll_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_leg_pelvic_yaw_joint",
    "right_leg_pelvic_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_leg_knee_pitch_joint",
    "right_leg_knee_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_leg_ankle_pitch_joint",
    "right_leg_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_leg_ankle_roll_joint",
    "right_leg_ankle_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


@ctrl_registry.register
class ParkourDiffusionCtrl(ControllerHook):
    """Upper-level Flow Matching reference generator for the Parkour policy."""

    def __init__(self, cfg_ctrl, env, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self.cfg_ctrl = cfg_ctrl
        self.env = env

        model_path = Path(cfg_ctrl.model_path).expanduser()
        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                "ParkourDiffusionCtrl only supports .onnx models; "
                f"got {model_path.suffix or '<no extension>'!r}: {model_path}"
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"Diffusion ONNX model not found: {model_path}")

        available = ort.get_available_providers()
        execution_provider = cfg_ctrl.execution_provider.lower()
        if execution_provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "casbot_diffusion_parkour requests ONNX CUDA inference, but "
                    f"CUDAExecutionProvider is unavailable. Available providers: {available}. "
                    "Install onnxruntime-gpu in the RoboJuDo CUDA environment."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif execution_provider == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(
                "execution_provider must be 'cuda' or 'cpu', "
                f"got {cfg_ctrl.execution_provider!r}"
            )
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(cfg_ctrl.onnx_intra_op_threads)
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=providers
        )
        logger.info(
            "Parkour Flow Matching ONNX providers: %s", self.session.get_providers()
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        required_metadata = {
            "diffusion.format_version",
            "diffusion.generative_method",
            "diffusion.sampling_steps",
            "diffusion.sampler",
            "diffusion.time_embedding_scale",
            "diffusion.q_low",
            "diffusion.q_high",
            "diffusion.t_q_low",
            "diffusion.t_q_high",
            "diffusion.p_q_low",
            "diffusion.p_q_high",
            "diffusion.history_size",
            "diffusion.future_size",
            "diffusion.future_frame_stride",
            "diffusion.root_body",
            "diffusion.terrain_layout",
            "diffusion.joint_layout",
            "diffusion.motion_layout",
            "diffusion.proprio_layout",
        }
        missing_metadata = required_metadata.difference(metadata)
        if missing_metadata:
            raise ValueError(
                "Diffusion ONNX is missing embedded metadata: "
                f"{sorted(missing_metadata)}. Re-export it with diffusion/source/utils/export.py."
            )
        if metadata["diffusion.format_version"] != "6":
            raise ValueError(
                "Casbot Parkour requires Flow Matching ONNX format_version=6; got "
                f"{metadata['diffusion.format_version']!r}. Retrain and re-export the model."
            )
        if metadata["diffusion.generative_method"] != "flow_matching":
            raise ValueError(
                "ParkourDiffusionCtrl no longer accepts DDPM checkpoints; got "
                f"generative_method={metadata['diffusion.generative_method']!r}."
            )
        if metadata["diffusion.root_body"] != "waist_yaw_link":
            raise ValueError(
                "Unexpected diffusion root body: "
                f"{metadata['diffusion.root_body']!r}"
            )
        if metadata["diffusion.terrain_layout"] != "root_z_minus_terrain_z":
            raise ValueError(
                "Unexpected diffusion terrain layout: "
                f"{metadata['diffusion.terrain_layout']!r}"
            )
        if metadata.get("diffusion.joint_layout") != "isaaclab_articulation":
            raise ValueError(
                "Unexpected diffusion joint layout: "
                f"{metadata.get('diffusion.joint_layout')!r}"
            )
        if metadata.get("diffusion.motion_layout") != "future_h0_heading_root_xyz_offset":
            raise ValueError(
                "Unexpected diffusion motion layout: "
                f"{metadata.get('diffusion.motion_layout')!r}"
            )
        if metadata.get("diffusion.proprio_layout") != (
            "joint_pos,root_velocity_local,velocity_command_local"
        ):
            raise ValueError(
                "Unexpected diffusion proprio layout: "
                f"{metadata.get('diffusion.proprio_layout')!r}"
            )

        input_shapes = {item.name: item.shape for item in self.session.get_inputs()}
        required_inputs = {"x_t", "timestep", "terrain", "proprio"}
        if set(input_shapes) != required_inputs:
            raise ValueError(
                f"Unexpected diffusion ONNX inputs: {input_shapes}; "
                f"expected {sorted(required_inputs)}"
            )
        self.window_size = int(input_shapes["x_t"][1])
        self.feature_dim = int(input_shapes["x_t"][2])
        self.history_size = int(metadata["diffusion.history_size"])
        self.future_size = int(metadata["diffusion.future_size"])
        self.future_frame_stride = int(metadata["diffusion.future_frame_stride"])
        self.dense_future_size = self.future_size * self.future_frame_stride
        if (
            self.window_size,
            self.feature_dim,
            self.history_size,
            self.future_size,
        ) != (10, 80, 4, 10):
            raise ValueError(
                "Casbot Parkour Flow Matching expects W=10, F=80, H=4, P=10; "
                f"got W={self.window_size}, F={self.feature_dim}, "
                f"H={self.history_size}, P={self.future_size}"
            )
        if self.future_frame_stride <= 0:
            raise ValueError(
                f"Invalid diffusion.future_frame_stride={self.future_frame_stride}"
            )
        logger.info(
            "Parkour Flow Matching horizon: %d keyframes x stride %d = %d control frames",
            self.future_size,
            self.future_frame_stride,
            self.dense_future_size,
        )

        self.sampler = FlowMatchingSampler(
            sampling_steps=int(metadata["diffusion.sampling_steps"]),
            method=metadata["diffusion.sampler"],
            time_embedding_scale=float(metadata["diffusion.time_embedding_scale"]),
        )

        for name in (
            "q_low", "q_high", "t_q_low", "t_q_high", "p_q_low",
            "p_q_high",
        ):
            values = json.loads(metadata[f"diffusion.{name}"])
            setattr(self, name, np.asarray(values, dtype=np.float32))

        self._env_to_diffusion = np.asarray(
            [self.env.joint_names.index(name) for name in DIFFUSION_JOINT_NAMES],
            dtype=np.int64,
        )

        self.velocity_command = np.asarray(cfg_ctrl.default_velocity_command, dtype=np.float32)
        self.reference_buffer: deque[dict[str, np.ndarray]] = deque()
        self.proprio_dim = int(input_shapes["proprio"][2])
        if self.proprio_dim != 31:
            raise ValueError(f"Expected proprio dim 31, got {self.proprio_dim}")
        self.joint_history = np.zeros((self.history_size, 25), dtype=np.float32)
        self.root_velocity_history = np.zeros((self.history_size, 3), dtype=np.float32)
        self.yaw_history = np.zeros(self.history_size, dtype=np.float32)
        self.root_pos_history = np.zeros((self.history_size, 3), dtype=np.float32)
        self.terrain_history = np.zeros((self.history_size, 693), dtype=np.float32)
        self._history_initialized = False
        self._previous_torso_pos: np.ndarray | None = None
        self.last_reference = np.zeros(65, dtype=np.float32)
        self.buffer_underflows = 0
        self.generated_windows = 0
        self.rng = np.random.default_rng(cfg_ctrl.seed)
        self.reset()

    @staticmethod
    def _normalize(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return 2.0 * (x - low) / (high - low) - 1.0

    def _generate(
        self,
        proprio_raw: np.ndarray,
        terrain_raw: np.ndarray,
    ) -> np.ndarray:
        terrain = np.asarray(terrain_raw, dtype=np.float32).reshape(
            1, self.history_size, 693
        )
        proprio = np.asarray(proprio_raw, dtype=np.float32).reshape(
            1, self.history_size, self.proprio_dim
        )
        terrain = self._normalize(terrain, self.t_q_low, self.t_q_high)
        proprio = self._normalize(proprio, self.p_q_low, self.p_q_high)

        x_t = self.rng.standard_normal(
            (1, self.window_size, self.feature_dim), dtype=np.float32
        )
        def velocity_fn(state: np.ndarray, model_time: np.ndarray) -> np.ndarray:
            return self.session.run(
                ["velocity"],
                {
                    "x_t": np.ascontiguousarray(state, dtype=np.float32),
                    "timestep": model_time,
                    "terrain": np.ascontiguousarray(terrain, dtype=np.float32),
                    "proprio": np.ascontiguousarray(proprio, dtype=np.float32),
                },
            )[0]

        x_t = self.sampler.sample(velocity_fn, x_t)

        motion = (
            (x_t.squeeze(0) + 1.0) * 0.5 * (self.q_high - self.q_low)
            + self.q_low
        )
        return motion.astype(np.float32)

    @staticmethod
    def _rotation_from_6d(value: np.ndarray) -> Rotation:
        col0 = np.asarray(value[:3], dtype=np.float64)
        col0 /= max(np.linalg.norm(col0), 1.0e-8)
        col2 = np.asarray(value[3:6], dtype=np.float64)
        col2 -= col0 * np.dot(col0, col2)
        col2 /= max(np.linalg.norm(col2), 1.0e-8)
        col1 = np.cross(col2, col0)
        return Rotation.from_matrix(np.stack((col0, col1, col2), axis=-1))

    @staticmethod
    def _rotation_to_6d(rotation: Rotation) -> np.ndarray:
        matrix = rotation.as_matrix()
        return np.concatenate((matrix[:, 0], matrix[:, 2])).astype(np.float32)

    def _decode_root(self, frame: np.ndarray) -> tuple[np.ndarray, Rotation]:
        h0_yaw = Rotation.from_euler("z", float(self.yaw_history[0]))
        root_pos = self.root_pos_history[0] + h0_yaw.apply(frame[:3])
        root_rot = h0_yaw * self._rotation_from_6d(frame[3:9])
        return root_pos.astype(np.float32), root_rot

    def _reference_to_command(self, reference: dict[str, np.ndarray]) -> np.ndarray:
        """Build the 65-D lower-policy command against the live robot pose."""
        root_pos = reference["root_pos"]
        root_rot = Rotation.from_quat(reference["root_quat"])
        robot_pos = np.asarray(self.env.torso_pos, dtype=np.float32)
        robot_rot = Rotation.from_quat(
            np.asarray(self.env.torso_quat, dtype=np.float64)
        )
        robot_inv = robot_rot.inv()
        root_pos_error_b = robot_inv.apply(root_pos - robot_pos)
        root_rot_error_b = robot_inv * root_rot

        target_lin_vel_b = robot_inv.apply(reference["root_lin_vel"])
        target_ang_vel_b = robot_inv.apply(reference["root_ang_vel"])
        return np.concatenate(
            (
                root_pos_error_b,
                self._rotation_to_6d(root_rot_error_b),
                target_lin_vel_b,
                target_ang_vel_b,
                reference["joint_pos"],
                reference["joint_vel"],
            )
        ).astype(np.float32)

    def _densify_motion(self, sparse_motion: np.ndarray) -> np.ndarray:
        """Interpolate actual H3 -> H3+5...H3+50 in model feature space."""
        dense = np.empty(
            (self.dense_future_size, self.feature_dim), dtype=np.float32
        )
        h0_yaw = Rotation.from_euler("z", float(self.yaw_history[0]))
        robot_pos = np.asarray(self.env.torso_pos, dtype=np.float32)
        robot_rot = Rotation.from_quat(
            np.asarray(self.env.torso_quat, dtype=np.float64)
        )
        previous_pos = h0_yaw.inv().apply(
            robot_pos - self.root_pos_history[0]
        ).astype(np.float32)
        previous_rot = h0_yaw.inv() * robot_rot
        previous_joint_pos = np.asarray(self.env.dof_pos, dtype=np.float32)[
            self._env_to_diffusion
        ]
        previous_joint_vel = np.asarray(self.env.dof_vel, dtype=np.float32)[
            self._env_to_diffusion
        ]
        previous_aux = sparse_motion[0, 59:].copy()

        for key_index in range(self.future_size):
            target = sparse_motion[key_index]
            target_pos = target[:3]
            target_rot = self._rotation_from_6d(target[3:9])
            target_joint_pos = target[9:34]
            target_joint_vel = target[34:59]
            target_aux = target[59:]
            relative_rotvec = (previous_rot.inv() * target_rot).as_rotvec()
            segment_start = key_index * self.future_frame_stride
            for substep in range(1, self.future_frame_stride + 1):
                alpha = substep / float(self.future_frame_stride)
                dense_index = segment_start + substep - 1
                dense[dense_index, :3] = (
                    (1.0 - alpha) * previous_pos + alpha * target_pos
                )
                interpolated_rot = previous_rot * Rotation.from_rotvec(
                    alpha * relative_rotvec
                )
                dense[dense_index, 3:9] = self._rotation_to_6d(
                    interpolated_rot
                )
                dense[dense_index, 9:34] = (
                    (1.0 - alpha) * previous_joint_pos
                    + alpha * target_joint_pos
                )
                dense[dense_index, 34:59] = (
                    (1.0 - alpha) * previous_joint_vel
                    + alpha * target_joint_vel
                )
                dense[dense_index, 59:] = (
                    (1.0 - alpha) * previous_aux + alpha * target_aux
                )
            previous_pos = target_pos
            previous_rot = target_rot
            previous_joint_pos = target_joint_pos
            previous_joint_vel = target_joint_vel
            previous_aux = target_aux
        return dense

    def _motion_to_references(
        self, dense_motion: np.ndarray
    ) -> list[dict[str, np.ndarray]]:
        """Decode a dense feature block into fixed world-frame references."""
        root_positions: list[np.ndarray] = []
        root_rotations: list[Rotation] = []
        for frame in dense_motion:
            root_pos, root_rot = self._decode_root(frame)
            root_positions.append(root_pos)
            root_rotations.append(root_rot)

        dt = float(self.env.control_dt)
        references: list[dict[str, np.ndarray]] = []
        for index, frame in enumerate(dense_motion):
            if index == self.dense_future_size - 1:
                other = index - 1
                lin_vel = (root_positions[index] - root_positions[other]) / dt
                delta_rot = root_rotations[index] * root_rotations[other].inv()
            else:
                other = index + 1
                lin_vel = (root_positions[other] - root_positions[index]) / dt
                delta_rot = root_rotations[other] * root_rotations[index].inv()
            ang_vel = delta_rot.as_rotvec() / dt
            references.append(
                {
                    "root_pos": root_positions[index].astype(np.float32),
                    "root_quat": root_rotations[index].as_quat().astype(np.float32),
                    "root_lin_vel": np.asarray(lin_vel, dtype=np.float32),
                    "root_ang_vel": np.asarray(ang_vel, dtype=np.float32),
                    # Flow Matching and the lower Parkour policy both use the
                    # IsaacLab articulation order.  Do not remap this boundary.
                    "joint_pos": frame[9:34].astype(np.float32),
                    "joint_vel": frame[34:59].astype(np.float32),
                }
            )
        return references

    @staticmethod
    def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _base_yaw(self, env_data=None) -> float:
        # Diffusion training uses waist_yaw_link (IsaacLab BFS body index 3)
        # as its root, so deployment heading must come from the same body.
        source = self.env.torso_quat if env_data is None else env_data["torso_quat"]
        if source is None:
            raise ValueError("ParkourDiffusionCtrl requires waist_yaw_link quaternion")
        return float(quatToEuler(np.asarray(source, dtype=np.float32))[2])

    def _proprio_condition(self) -> np.ndarray:
        repeated_command = np.broadcast_to(
            self.velocity_command, (self.history_size, 3)
        )
        return np.concatenate(
            [self.joint_history, self.root_velocity_history, repeated_command], axis=-1
        )

    def _append_history(self, env_data) -> None:
        joint_pos = np.asarray(env_data["dof_pos"], dtype=np.float32)[
            self._env_to_diffusion
        ]
        yaw = self._base_yaw(env_data)
        terrain = self._terrain_z(env_data)
        root_pos = np.asarray(env_data["torso_pos"], dtype=np.float32)
        if self._previous_torso_pos is None:
            local_lin_xy = np.zeros(2, dtype=np.float32)
        else:
            world_lin = (root_pos - self._previous_torso_pos) / float(self.env.control_dt)
            local_lin_xy = Rotation.from_euler("z", yaw).inv().apply(world_lin)[:2].astype(np.float32)
        torso_ang_vel = env_data.get("torso_ang_vel")
        local_wz = 0.0 if torso_ang_vel is None else float(np.asarray(torso_ang_vel)[2])
        root_velocity = np.asarray([local_lin_xy[0], local_lin_xy[1], local_wz], dtype=np.float32)
        self._previous_torso_pos = root_pos.copy()
        if not self._history_initialized:
            self.joint_history[:] = joint_pos
            self.root_velocity_history[:] = root_velocity
            self.yaw_history[:] = yaw
            self.terrain_history[:] = terrain
            self.root_pos_history[:] = root_pos
            self._history_initialized = True
            return
        self.joint_history[:-1] = self.joint_history[1:]
        self.joint_history[-1] = joint_pos
        self.root_velocity_history[:-1] = self.root_velocity_history[1:]
        self.root_velocity_history[-1] = root_velocity
        self.yaw_history[:-1] = self.yaw_history[1:]
        self.yaw_history[-1] = yaw
        self.terrain_history[:-1] = self.terrain_history[1:]
        self.terrain_history[-1] = terrain
        self.root_pos_history[:-1] = self.root_pos_history[1:]
        self.root_pos_history[-1] = root_pos

    def _handle_keyboard(self, prior_ctrl_data) -> None:
        keyboard = prior_ctrl_data.get("KeyboardCtrl")
        if keyboard is None:
            return
        command_changed = False
        for event in keyboard.get("keyboard_event", []):
            if event.get("type") != "keyboard" or event.get("pressed"):
                continue
            key = event.get("name")
            if key == "Key.left":
                self.velocity_command[2] += self.cfg_ctrl.angular_velocity_step
                command_changed = True
            elif key == "Key.right":
                self.velocity_command[2] -= self.cfg_ctrl.angular_velocity_step
                command_changed = True
            elif key == "Key.up":
                self.velocity_command[0] += self.cfg_ctrl.linear_velocity_step
                command_changed = True
            elif key == "Key.down":
                self.velocity_command[0] -= self.cfg_ctrl.linear_velocity_step
                command_changed = True
            elif key == "1":
                self.velocity_command[1] += self.cfg_ctrl.linear_velocity_step
                command_changed = True
            elif key == "2":
                self.velocity_command[1] -= self.cfg_ctrl.linear_velocity_step
                command_changed = True
            elif key == "Key.space":
                self.velocity_command[:] = 0.0
                command_changed = True

        limits = np.asarray(self.cfg_ctrl.velocity_command_limits, dtype=np.float32)
        self.velocity_command = np.clip(self.velocity_command, -limits, limits)
        if command_changed:
            print(f"[ParkourDiffusionCtrl] velocity command [vx, vy, wz]: {self.velocity_command}")

    @staticmethod
    def _terrain_z(env_data) -> np.ndarray:
        height_scan = env_data.get("height_scan")
        if height_scan is None:
            raise ValueError("ParkourDiffusionCtrl requires a 33x21 XYZ height scan")
        scan = np.asarray(height_scan, dtype=np.float32).reshape(-1, 3)
        if scan.shape[0] != 693:
            raise ValueError(f"Expected 693 height-scan points, got {scan.shape[0]}")
        # MujocoEnv height_scan z is terrain_z - waist_z, matching the lower
        # Parkour policy.  Diffusion K/V uses the opposite-sign clearance:
        # waist_z - terrain_z.
        return (-scan[:, 2]).astype(np.float32)

    def _generate_dense_block(self) -> None:
        """Generate ten sparse keyframes and enqueue all 50 dense references."""
        try:
            sparse_window = self._generate(
                self._proprio_condition(),
                self.terrain_history.copy(),
            )
            if not np.isfinite(sparse_window).all():
                raise ValueError("Flow Matching produced NaN/Inf")
            dense_window = self._densify_motion(sparse_window)
            self.reference_buffer.extend(
                self._motion_to_references(dense_window)
            )
            self.generated_windows += 1
        except Exception:
            logger.exception("Parkour Flow Matching generation failed; keeping previous reference")

    def get_data_with_hook(self, prior_ctrl_data: dict, env_data: dict):
        self._handle_keyboard(prior_ctrl_data)
        self._append_history(env_data)
        if not self.reference_buffer:
            self._generate_dense_block()

        if self.reference_buffer:
            reference = self.reference_buffer.popleft()
            frame = self._reference_to_command(reference)
            self.last_reference = frame.copy()
        else:
            self.buffer_underflows += 1
            print(
                "[WARN][ParkourDiffusionCtrl] reference buffer empty "
                f"(underflow={self.buffer_underflows}, "
                f"generated_windows={self.generated_windows}); "
                "reusing the previous reference frame",
                flush=True,
            )
            frame = self.last_reference
            reference = {
                "root_pos": np.asarray(self.env.torso_pos, dtype=np.float32),
                "root_quat": np.asarray(self.env.torso_quat, dtype=np.float32),
                "joint_pos": frame[15:40].copy(),
                "joint_vel": frame[40:65].copy(),
            }
        torso_pos = self.env.torso_pos
        torso_quat = self.env.torso_quat
        if torso_pos is None or torso_quat is None:
            raise ValueError("ParkourDiffusionCtrl requires torso pose from the environment")
        return {
            "command": frame.copy(),
            "joint_pos": frame[15:40].copy(),
            "joint_vel": frame[40:65].copy(),
            "robot_anchor_pos_w": torso_pos.copy(),
            "robot_anchor_quat_w": torso_quat.copy(),
            "anchor_pos_w": reference["root_pos"].copy(),
            "anchor_quat_w": reference["root_quat"].copy(),
            "velocity_command": self.velocity_command.copy(),
            "hand_pose": None,
        }

    def reset(self):
        env_pos = np.asarray(self.env.dof_pos, dtype=np.float32)[
            self._env_to_diffusion
        ]
        self.reference_buffer.clear()
        self.last_reference[:] = 0.0
        self.last_reference[15:40] = env_pos
        initial_joint_pos = np.asarray(self.env.dof_pos, dtype=np.float32)[
            self._env_to_diffusion
        ]
        initial_yaw = self._base_yaw()
        self.joint_history[:] = initial_joint_pos
        self.root_velocity_history[:] = 0.0
        self.yaw_history[:] = initial_yaw
        self.velocity_command[:] = np.asarray(
            self.cfg_ctrl.default_velocity_command, dtype=np.float32
        )
        self.terrain_history[:] = 0.0
        initial_root_pos = np.asarray(self.env.torso_pos, dtype=np.float32)
        self.root_pos_history[:] = initial_root_pos
        self._previous_torso_pos = None
        self._history_initialized = False
        self.buffer_underflows = 0
        self.generated_windows = 0

    def post_step_callback(self, commands: list[str] | None = None):
        if "[MOTION_RESET]" in (commands or []):
            self.reset()
