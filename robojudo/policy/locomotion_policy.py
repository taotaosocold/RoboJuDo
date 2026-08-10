import logging

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import LocomotionPolicyCfg
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


@policy_registry.register
class LocomotionPolicy(Policy):
    cfg_policy: LocomotionPolicyCfg

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(cfg_policy.policy_file, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.obs_scales = self.cfg_policy.obs_scales
        self.max_cmd = np.asarray(self.cfg_policy.max_cmd)
        self.command_step = self.cfg_policy.command_step
        self.action_scales = np.asarray(self.cfg_policy.action_scales)
        self.use_height_scan = self.cfg_policy.use_height_scan
        self.last_joystick_axes = np.zeros(3)
        self.last_velocity_commands = np.zeros(3)
        self.velocity_commands = np.zeros(3)
        self.command_axis_ready = np.ones(3, dtype=bool)

        input_size = self.session.get_inputs()[0].shape[-1]
        observation_size = 9 + 3 * self.num_dofs
        if self.use_height_scan:
            observation_size += 33 * 21 * 3
        if isinstance(input_size, int) and input_size != observation_size:
            raise ValueError(f"Policy input size is {input_size}, expected {observation_size}")

        self.reset()

    def reset(self):
        self.timestep: int = 0
        self.last_action = np.zeros(self.num_actions)
        self.velocity_commands = np.zeros(3)
        self.command_axis_ready = np.ones(3, dtype=bool)

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def _get_commands(self, ctrl_data):
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx, ly, rx = axes["LeftX"], axes["LeftY"], axes["RightX"]
                joystick_axes = np.array([lx, ly, rx])
                if not np.allclose(joystick_axes, self.last_joystick_axes, atol=0.05):
                    logger.info(f"[LocomotionPolicy] joystick_axes [LeftX, LeftY, RightX]: {joystick_axes}")
                    self.last_joystick_axes = joystick_axes.copy()

                command_axes = np.array([ly, -lx, -rx])
                for index, value in enumerate(command_axes):
                    if abs(value) < 0.2:
                        self.command_axis_ready[index] = True
                    elif abs(value) > 0.5 and self.command_axis_ready[index]:
                        self.velocity_commands[index] += np.sign(value) * self.command_step
                        self.command_axis_ready[index] = False
                self.velocity_commands = np.clip(self.velocity_commands, -self.max_cmd, self.max_cmd)
                break
            if key == "KeyboardCtrl":
                for event in ctrl_data[key]["keyboard_event"]:
                    if event["type"] != "keyboard" or not event["pressed"]:
                        continue
                    match event["name"]:
                        case "w":
                            self.velocity_commands[0] += self.command_step
                        case "s":
                            self.velocity_commands[0] -= self.command_step
                        case "a":
                            self.velocity_commands[1] += self.command_step
                        case "d":
                            self.velocity_commands[1] -= self.command_step
                        case "e":
                            self.velocity_commands[2] -= self.command_step
                        case "q":
                            self.velocity_commands[2] += self.command_step
                self.velocity_commands = np.clip(self.velocity_commands, -self.max_cmd, self.max_cmd)
                break
        return self.velocity_commands.copy()

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)
        velocity_commands = commands * self.obs_scales.command
        if not np.allclose(velocity_commands, self.last_velocity_commands, atol=0.01):
            logger.info(f"[LocomotionPolicy] velocity_commands: {velocity_commands}")
            self.last_velocity_commands = velocity_commands.copy()
        gravity_orientation = get_gravity_orientation(env_data.base_quat)

        observation_components = {
            "base_ang_vel": env_data.base_ang_vel * self.obs_scales.ang_vel,
            "projected_gravity": gravity_orientation,
            "velocity_commands": velocity_commands,
            "joint_pos": env_data.dof_pos - self.default_dof_pos,
            "joint_vel": env_data.dof_vel * self.obs_scales.dof_vel,
            "actions": self.last_action,
        }

        if self.use_height_scan:
            height_scan = env_data.get("height_scan")
            if height_scan is None:
                raise ValueError("height_scan is required when use_height_scan is True")
            observation_components["height_scan"] = height_scan

        obs = np.concatenate(list(observation_components.values())).astype(np.float32)
        extras = {
            "commands": commands,
            "observation_components": observation_components,
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {self.input_name: np.expand_dims(obs, axis=0).astype(np.float32)}
        actions = np.asarray(self.session.run(None, ort_inputs)[0]).squeeze()

        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        self.last_action = actions.copy()

        if self.action_clip is not None:
            actions = np.clip(actions, -self.action_clip, self.action_clip)

        return actions * self.action_scales
