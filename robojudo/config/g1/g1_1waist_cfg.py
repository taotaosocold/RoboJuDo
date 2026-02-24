from robojudo.config import cfg_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from .env.g1_mujuco_env_cfg import G1_23MujocoEnvCfg
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg

from .ctrl.g1_1waist_ctrl_cfg import G1VaeMimicCtrlCfg
from .policy.g1_1waist_policy_cfg import G1VaeMimicPolicyCfg

@cfg_registry.register
class g1_vaemimic(RlPipelineCfg):
    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1VaeMimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1VaeMimicCtrlCfg(
            motion_name = '90_08_poses'
        )
    ]
    policy: G1VaeMimicPolicyCfg = G1VaeMimicPolicyCfg(
        policy_name = "0221",
        use_motion_from_model=False,
        max_timestep=250
    )