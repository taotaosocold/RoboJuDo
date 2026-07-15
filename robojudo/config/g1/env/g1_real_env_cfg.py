from typing import Literal

from robojudo.environment.env_cfgs import UnitreeEnvCfg

# from robojudo.tools.tool_cfgs import ZedOdometryCfg
from .g1_env_cfg import G1EnvCfg


class G1UnitreeCfg(UnitreeEnvCfg.UnitreeCfg):
    robot: Literal["h1", "g1"] = "g1"

    msg_type: Literal["hg", "go"] = "hg"
    hand_type: Literal["Dex-3", "Inspire", "NONE"] = "NONE"

    enable_odometry: bool = True

# 如果走实机则几乎都是选择这个配置
class G1RealEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    # env_type: str = UnitreeEnvCfg.model_fields["env_type"].default
    # 默认的环境类名是UnitreeCppEnv而不是UnitreeEnv
    env_type: str = "UnitreeCppEnv"
    # ====== ENV CONFIGURATION ======
    unitree: UnitreeEnvCfg.UnitreeCfg = G1UnitreeCfg(
        net_if="eth0",
    )
    # 实机跑的时候，G1是没法直接测出身体在世界坐标系下的位置和线速度的，所以需要外部里程计来估算base_pos和base_lin_vel
    # 这里有四个选择，NONE就是不用，则base_pos和base_lin_vel始终为0，DUMMY是假里程计即返回固定值，UNITREE是宇树用SDK内部的状态估计，ZED是用双目相机作视觉里程计（SLAM）
    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "UNITREE"

    joint2motor_idx: list[int] | None = None  # list(range(0, 29))


class G1WithHandRealEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    # env_type: str = UnitreeEnvCfg.model_fields["env_type"].default
    env_type: str = "UnitreeCppEnv"
    # ====== ENV CONFIGURATION ======
    unitree: UnitreeEnvCfg.UnitreeCfg = G1UnitreeCfg(
        net_if="eth0",
        hand_type="Dex-3",
    )

    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "DUMMY"
    # zed_cfg: ZedOdometryCfg | None = ZedOdometryCfg(
    #     server_ip="192.168.123.167",
    #     pos_offset=[0.0, 0.0, 0.9],
    #     zero_align=True,
    # )

    joint2motor_idx: list[int] | None = None  # list(range(0, 29))
