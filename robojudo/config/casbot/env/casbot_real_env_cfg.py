from typing import Literal

from robojudo.config import Config
from robojudo.environment.env_cfgs import RobotEnvCfg

from .casbot_env_cfg import CasbotEnvCfg


class CasbotHlCfg(Config):
    """HL Robot ROS2 communication config."""

    joint_cmd_topic: str = "/motion/joint_cmd"
    joint_state_topic: str = "/joint_states"
    imu_topic: str = "/imu"
    upper_cmd_topic: str = "/upper/joint_cmd"

    control_dt: float = 0.02
    """50Hz control command interval for background send timer"""

    robot_joint_names: list[str] | None = None
    """Joint names used by the robot SDK (in same order as env joint_names).
    When set, used for both outgoing /motion/joint_cmd and incoming /motion/joint_state.
    When None, env joint_names are used directly (robot and RoboJuDo share the same naming)."""


class CasbotRealEnvCfg(CasbotEnvCfg, RobotEnvCfg):
    """
    Casbot real robot environment configuration using hl_motion ROS2 SDK.
    MRO: CasbotRealEnvCfg → CasbotEnvCfg → RobotEnvCfg → EnvCfg → Config
    """

    env_type: str = "CasbotRealEnv"

    hl: CasbotHlCfg = CasbotHlCfg()

    odometry_type: Literal["NONE", "DUMMY", "ZED"] = "DUMMY"

    joint2motor_idx: list[int] | None = None
    """Optional motor index mapping, None uses name-based matching"""
