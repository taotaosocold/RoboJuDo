from typing import Literal

from robojudo.environment.env_cfgs import CasbotHlCfg, RobotEnvCfg

from .casbot_env_cfg import CasbotEnvCfg


class CasbotRealEnvCfg(CasbotEnvCfg, RobotEnvCfg):
    """
    Casbot real robot environment configuration using hl_motion ROS2 SDK.
    MRO: CasbotRealEnvCfg → CasbotEnvCfg → RobotEnvCfg → EnvCfg → Config
    """

    env_type: str = "CasbotRealEnv"

    hl: CasbotHlCfg = CasbotHlCfg()

    odometry_type: Literal["NONE", "DUMMY", "ZED"] = "DUMMY"

    joint2motor_idx: list[int] | None = None
    """Mapping from env DOF index to robot command index.
    When None, auto-computed from hl.robot_joint_names via name matching."""
