import logging
from collections import deque

import numpy as np
import onnxruntime as ort

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import ParkourPolicyCfg
from robojudo.tools.dof import DoFConfig
from robojudo.utils.util_func import quat_rotate_inverse_np, subtract_frame_transforms

logger = logging.getLogger(__name__)


@policy_registry.register
class ParkourPolicy(Policy):
    cfg_policy: ParkourPolicyCfg

    def __init__(self, cfg_policy: ParkourPolicyCfg, device):
        # init onnx, override dof cfg if needed
        sess_options = ort.SessionOptions()
        # Avoid the lower policy and asynchronous diffusion session each
        # creating an unrestricted CPU thread pool.
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        device = "cpu"
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            # Jetson
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError(f"Unknown device: {device}")

        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        cfg_policy_new = cfg_policy.model_copy()
        if cfg_policy_new.use_modelmeta_config:
            logger.info("[ParkourPolicy] Using modelmeta as config ...")
            modelmeta = self.session.get_modelmeta()  # all str,
            modelmeta_dict = modelmeta.custom_metadata_map

            # dict_keys(['joint_names', 'run_path', 'command_names', 'joint_stiffness', 'joint_damping',
            # 'default_joint_pos', 'action_scale', 'observation_names', 'observation_history_lengths'])
            def parse_floats(s):
                return [float(item) for item in s.split(",")]

            def parse_strings(s):
                return [item for item in s.split(",")]

            dof_config = DoFConfig(
                joint_names=parse_strings(modelmeta_dict["joint_names"]),
                default_pos=parse_floats(modelmeta_dict["default_joint_pos"]),
                stiffness=parse_floats(modelmeta_dict["joint_stiffness"]),
                damping=parse_floats(modelmeta_dict["joint_damping"]),
            )
            action_scales = parse_floats(modelmeta_dict["action_scale"])

            cfg_policy_new.action_dof = dof_config
            cfg_policy_new.obs_dof = dof_config
            cfg_policy_new.action_scales = action_scales

            observation_names = parse_strings(modelmeta_dict["observation_names"])
            history_lengths = modelmeta_dict.get("observation_history_lengths")
            if history_lengths is None:
                history_lengths = [1] * len(observation_names)
            else:
                history_lengths = [int(float(item)) for item in history_lengths.split(",")]
        else:
            observation_names = list(cfg_policy_new.observation_names)
            history_lengths = list(cfg_policy_new.observation_history_lengths)

        super().__init__(cfg_policy=cfg_policy_new, device=device)
        self.action_scales = np.asarray(self.cfg_policy.action_scales)

        self.observation_names = observation_names
        if len(history_lengths) != len(self.observation_names):
            raise ValueError(
                "Parkour ONNX metadata has mismatched observation_names and "
                f"observation_history_lengths: {len(self.observation_names)} != {len(history_lengths)}"
            )
        self.observation_history_lengths = dict(
            zip(self.observation_names, history_lengths, strict=True)
        )

        self.command = None
        self.history_buffers = {}
        self.reset()

    def _prepare_policy(self):
        obs_shape = self.session.get_inputs()[0].shape  # e.g. [1, 154]
        obs = np.zeros(obs_shape[1], dtype=np.float32)
        self.get_action(obs)

    def reset(self):
        self.timestep: float = 0
        self.pbar = None
        self.play_speed: float = 1.0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.history_buffers.clear()
        self.flag_motion_done = False
        self._prepare_policy()
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1 * self.play_speed
        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.play_speed = 1.0
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0

    def _get_command(self, env_data, ctrl_data):
        del env_data  # Parkour always loads its reference motion from the controller.
        source = self.cfg_policy.command_source
        if source not in ctrl_data:
            raise KeyError(f"Parkour command source {source!r} not found in ctrl_data")
        command = ctrl_data.get(source)
        self.command = command
        # print(command.time_steps[0])
        return (
            command.command,
            command.robot_anchor_pos_w,
            command.robot_anchor_quat_w,
            command.anchor_pos_w,
            command.anchor_quat_w,
            command.get("hand_pose", None),
        )

    def _update_observation_history(self, observation_components):
        history_components = {}
        for name in self.observation_names:
            value = np.asarray(observation_components[name], dtype=np.float32).reshape(-1)
            history_length = self.observation_history_lengths[name]
            history_buffer = self.history_buffers.get(name)
            if history_buffer is None:
                # Match IsaacLab CircularBuffer: fill all slots with the first frame after reset.
                history_buffer = deque(
                    (value.copy() for _ in range(history_length)),
                    maxlen=history_length,
                )
                self.history_buffers[name] = history_buffer
            else:
                history_buffer.append(value.copy())
            history_components[name] = np.concatenate(history_buffer)
        return history_components

    def get_observation(self, env_data, ctrl_data):
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        base_quat = env_data.base_quat
        ang_vel = env_data.base_ang_vel
        height_scan = env_data.get("height_scan")
        if height_scan is None:
            raise ValueError("ParkourPolicy requires a 33x21 XYZ height scan")

        command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, hand_pose = (
            self._get_command(env_data, ctrl_data)
        )

        pos, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        obs_command = command
        obs_motion_anchor_pos_b = pos
        obs_projected_gravity = quat_rotate_inverse_np(
            base_quat,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        obs_base_ang_vel = ang_vel
        obs_joint_pos_rel = dof_pos - self.default_dof_pos
        obs_joint_vel_rel = dof_vel
        obs_last_action = self.last_action
        obs_height_scan = height_scan

        observation_components = {
            "command": obs_command,
            "motion_anchor_pos_b": obs_motion_anchor_pos_b,
            "projected_gravity": obs_projected_gravity,
            "base_ang_vel": obs_base_ang_vel,
            "joint_pos": obs_joint_pos_rel,
            "joint_vel": obs_joint_vel_rel,
            "actions": obs_last_action,
            "height_scan": obs_height_scan,
        }
        history_components = self._update_observation_history(observation_components)
        obs = np.concatenate([history_components[name] for name in self.observation_names]).astype(np.float32)

        model_size = self.session.get_inputs()[0].shape[-1]
        if isinstance(model_size, int) and obs.size != model_size:
            component_sizes = {name: value.size for name, value in history_components.items()}
            raise ValueError(
                f"Parkour observation has {obs.size} values, model expects {model_size}: {component_sizes}"
            )

        extras = {
            "pos": pos,
            "ori": ori,
            "robot_anchor_pos_w": robot_anchor_pos_w,
            "robot_anchor_quat_w": robot_anchor_quat_w,
            "anchor_pos_w": anchor_pos_w,
            "anchor_quat_w": anchor_quat_w,
            "command": command,
            "observation_components": history_components,
            "hand_pose": hand_pose,
            "CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else [],
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {
            self.input_names[0]: np.expand_dims(obs, axis=0).astype(np.float32),
        }
        ort_outputs = self.session.run([self.output_names[0]], ort_inputs)
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        self.last_action = actions.copy()

        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)
        return actions * self.action_scales

    def get_init_dof_pos(self) -> np.ndarray:
        """Return the first frame of the external reference motion when available."""
        if self.command is not None:
            joint_pos = self.command.get("joint_pos", None)
            if joint_pos is not None:
                return np.asarray(joint_pos).copy()
        return self.default_dof_pos.copy()

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        robot_anchor_pos_w = extras["robot_anchor_pos_w"]
        robot_anchor_quat_w = extras["robot_anchor_quat_w"]
        anchor_pos_w = extras["anchor_pos_w"]
        anchor_quat_w = extras["anchor_quat_w"]

        pos = extras["pos"]
        # ori = extras["ori"]

        visualizer.draw_arrow(anchor_pos_w, anchor_quat_w, [0.2, 0, 0], color=[1, 0, 0, 1], scale=2, id=0)
        visualizer.draw_arrow(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            [0.2, 0, 0],
            color=[0, 1, 0, 1],
            scale=2,
            id=1,
        )
        visualizer.draw_arrow(robot_anchor_pos_w, robot_anchor_quat_w, pos, color=[0, 1, 1, 1], scale=2, id=2)

        torso_pos = env_data["torso_pos"]
        torso_quat = env_data["torso_quat"]

        visualizer.draw_arrow(torso_pos, torso_quat, [0.2, 0, 0], color=[1, 1, 0, 1], scale=2, id=3)
