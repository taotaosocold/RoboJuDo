import logging
import time
from enum import Enum

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class RobotState(Enum):
    PASSIVE = "passive"
    FIXED_STAND = "fixed_stand"
    POLICY_CONTROL = "policy_control"
    ESTOP = "estop"


class PolicyWrapper:
    """A wrapper for Policy to handle observation and action adaptation."""

    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.env_dof_cfg = env_dof_cfg

        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name  # type: ignore
        # while policy_name in self.policies.keys():
        #     policy_name += "_new"
        self.name = policy_name

        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.actions_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

    def get_observation(self, env_data: Box, ctrl_data: Box):
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)
        return self.policy.get_observation(env_data_adapted, ctrl_data)

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return self.actions_adapter.fit(action)

    def get_pd_target(self, obs, motion_joint_pos: np.ndarray | None = None):
        action = self.policy.get_action(obs)
        use_residual = getattr(self.policy.cfg_policy, "use_residual_action", False)
        if use_residual and motion_joint_pos is not None:
            # pd_target = action * scale + motion_joint_pos (residual on reference motion)
            # note: action_scales already applied inside get_action(), so action is already scaled
            motion_joint_pos_adapted = self.obs_adapter.fit(motion_joint_pos)
            pd_target = action + motion_joint_pos_adapted
        else:
            pd_target = action + self.policy.default_pos
        return self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)

    def get_init_dof_pos(self):
        return self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)

    def __getattr__(self, name):
        """Fallback: delegate other func to the wrapped policy."""
        return getattr(self.policy, name)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        self.policy = PolicyWrapper(
            cfg_policy=self.cfg.policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,
        )

        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        # ---- FSM ----
        self._fsm_enabled = self.cfg.fsm_enabled
        self._state = RobotState.PASSIVE
        self._stand_start_dof_pos = None
        self._transition_step = 0
        self._stand_target_pos = (
            np.array(self.cfg.stand_target_pos)
            if self.cfg.stand_target_pos is not None
            else self.policy.default_pos
        )

        self.self_check()
        self.reset()

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0

        self.env.reset()
        # self.env.reborn(init_qpos=[0.2, 0.2, 0.8] + [ 0.707, 0, 0, 0.707]) # FOR SIM DEBUG
        self.policy.reset()
        self.ctrl_manager.reset()

    def safety_check(self):
        if not self.do_safety_check:
            return
        # check joint velocity
        if self.cfg.dof_vel_limit is not None:
            max_vel = np.max(np.abs(self.env.dof_vel))
            if max_vel > self.cfg.dof_vel_limit:
                logger.error(f"Joint velocity {max_vel:.1f} rad/s exceeds limit {self.cfg.dof_vel_limit}! Shutdown.")
                self._shutdown_or_reborn()
                return
        # check robot orientation
        gravity_ori = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
        if abs(angle) > 1.0:  # more than ~57 degrees
            logger.error("Robot fallen! Shutdown for safety.")
            self._shutdown_or_reborn()

    def _shutdown_or_reborn(self):
        if hasattr(self.env, "reborn"):
            self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()

        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")
        # 状态机的切换放在策略执行之前，这样如果不是POLICY_CONTROL即策略控制下是不会走下面的代码的
        if self._fsm_enabled:
            self._fsm_handle_commands(commands)
            if self._state != RobotState.POLICY_CONTROL:
                if dry_run:
                    self.policy.get_observation(env_data, ctrl_data)
                self._fsm_non_policy_step(env_data, ctrl_data, dry_run)
                return

        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        # For residual action mode: extract motion reference joint_pos from ctrl_data
        beyondmimic_ctrl_data = ctrl_data.get("BeyondMimicCtrl", None)
        motion_joint_pos = beyondmimic_ctrl_data.get("joint_pos", None) if beyondmimic_ctrl_data is not None else None
        pd_target = self.policy.get_pd_target(obs, motion_joint_pos=motion_joint_pos)

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

    # ---- FSM methods ----

    def _fsm_handle_commands(self, commands: list[str]):
        # 从硬件中读取我们是想要切到哪种状态，command会给出比如[STATE_PASSIVE]的期望状态
        for command in commands:
            match command:
                # 会通过_fsm_transition对切入不同的状态做不同的预处理
                case "[STATE_PASSIVE]":
                    self._fsm_transition(RobotState.PASSIVE)
                case "[STATE_FIXED_STAND]":
                    self._fsm_transition(RobotState.FIXED_STAND)
                case "[STATE_POLICY]":
                    self._fsm_transition(RobotState.POLICY_CONTROL)
                case "[STATE_ESTOP]" | "[SHUTDOWN]":
                    self._fsm_transition(RobotState.ESTOP)
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]

    def _fsm_transition(self, new_state: RobotState):
        # 如果已经在目标状态就直接返回
        if new_state == self._state:
            return
        logger.info(f"[FSM] {self._state.value} → {new_state.value}")
        self._state = new_state
        # 如果是进入PASSIVE或FIXED_STAND状态这里会对_stand_start_dof_pos和_stand_start_dof_pos重新赋值
        if new_state in (RobotState.PASSIVE, RobotState.FIXED_STAND):
            self._stand_start_dof_pos = self.env.dof_pos.copy()
            self._transition_step = 0
            if new_state == RobotState.FIXED_STAND:
                print("=" * 80)
                print(
                    "[FSM] FIXED_STAND target, "
                    f"stand_transition_steps={self.cfg.stand_transition_steps}"
                )
                for i, name in enumerate(self.env.joint_names):
                    cur = self._stand_start_dof_pos[i]
                    tgt = self._stand_target_pos[i]
                    print(f"  [{i:2d}] {name:40s} current={cur: 8.4f}  target={tgt: 8.4f}")
                print("=" * 80)
        # 如果是进入策略，那么timestep会清0，重置policy等
        elif new_state == RobotState.POLICY_CONTROL:
            self.reset()
    # 走非策略的代码比如passive或者fixedstand
    def _step_interpolate(self, target_pos: np.ndarray, transition_steps: int):
        """Linearly interpolate to target_pos in an exact number of control steps."""
        if transition_steps <= 0:
            raise ValueError(f"transition_steps must be positive, got {transition_steps}")
        if self._stand_start_dof_pos is None:
            self._stand_start_dof_pos = self.env.dof_pos.copy()
            self._transition_step = 0
        self._transition_step += 1
        alpha = min(self._transition_step / transition_steps, 1.0)
        target = (1 - alpha) * self._stand_start_dof_pos + alpha * target_pos
        self.env.step(target)

    def _fsm_non_policy_step(self, env_data, ctrl_data, dry_run: bool):
        if self._state == RobotState.PASSIVE:
            if not dry_run:
                self._step_interpolate(
                    np.zeros(self.env.num_dofs), self.cfg.passive_transition_steps
                )
        elif self._state == RobotState.FIXED_STAND:
            if not dry_run:
                self._step_interpolate(
                    self._stand_target_pos, self.cfg.stand_transition_steps
                )
        elif self._state == RobotState.ESTOP:
            if not dry_run:
                self.env.step(self.policy.default_pos)
            self._do_estop()
            return

        self._fsm_post_step(env_data, ctrl_data)

    def _do_estop(self):
        logger.warning("[FSM] Emergency stop!")
        if hasattr(self.env, "reborn"):
            self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.env.shutdown()
        self._state = RobotState.PASSIVE

    def _fsm_post_step(self, env_data, ctrl_data):
        """Minimal post-step for non-policy states: timestep, ctrl update, safety check."""
        self.timestep += 1
        self.ctrl_manager.post_step_callback(ctrl_data)
        self.safety_check()

    def prepare(self, init_motor_angle=None):
        if init_motor_angle is not None:
            desired_motor_angle = init_motor_angle
        else:
            desired_motor_angle = self.policy.get_init_dof_pos()

        # logger.info(f"{desired_motor_angle=}")
        current_motor_angle = np.array(self.env.dof_pos)
        # logger.info(f"{current_motor_angle=}")

        traj_len = 1000
        last_step_time = time.time()
        logger.warning("prepare_init")
        pbar = ProgressBar("Prepare", traj_len)

        for t in range(traj_len):
            current_motor_angle = np.array(self.env.dof_pos)

            blend_ratio = np.minimum(t / 300, 1)
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle

            # warm up network
            self.step(dry_run=True)

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()

            if t == 0.9 * traj_len:
                logger.info(f"{'=' * 10} RESET ZERO POSITION {'=' * 10}")
                self.reset()

        time.sleep(0.01)
        pbar.close()
        logger.warning("prepare_done")


if __name__ == "__main__":
    pass
