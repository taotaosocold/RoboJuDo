from robojudo.policy.policy_cfgs import LocomotionPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from .casbot_beyondmimic_policy_cfg import CasbotBeyondMimicDoF


class CasbotLocomotionPolicyCfg(LocomotionPolicyCfg):
    robot: str = "casbot"
    policy_name: str = "policy"

    obs_dof: DoFConfig = CasbotBeyondMimicDoF()
    action_dof: DoFConfig = obs_dof

    use_height_scan: bool = True
    action_beta: float = 1.0
    action_scales: list[float] = [
        0.136, 0.136, 0.054, 0.136, 0.136,
        0.0, 0.144, 0.144, 0.096, 0.096,
        0.0, 0.144, 0.144, 0.136, 0.136,
        0.093, 0.093, 0.096, 0.096, 0.144,
        0.144, 0.096, 0.096, 0.093, 0.093,
    ]
