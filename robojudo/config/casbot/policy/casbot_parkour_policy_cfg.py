import os

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import ParkourPolicyCfg

from .casbot_beyondmimic_policy_cfg import CasbotBeyondMimicDoF


class CasbotParkourPolicyCfg(ParkourPolicyCfg):
    """CASBOT policy exported from Tracking-Parkour-CASBOT-v0."""

    robot: str = "casbot"
    policy_name: str = "policy"

    @property
    def policy_file(self) -> str:
        return os.environ.get(
            "CASBOT_PARKOUR_POLICY_ONNX",
            "/home/casbot/Desktop/whole_body_tracking/logs/rsl_rl/CASBOT_parkour_diffusion/2026-08-18_23-54-09/exported/policy.onnx",
        )

    obs_dof: CasbotBeyondMimicDoF = CasbotBeyondMimicDoF()
    action_dof: CasbotBeyondMimicDoF = obs_dof

    use_modelmeta_config: bool = True
    action_beta: float = 1.0

    # Fallback layout used when modelmeta is disabled or absent.
    observation_names: list[str] = [
        "command",
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
        "height_scan",
    ]
    observation_history_lengths: list[int] = [1, 1, 1, 1, 1, 1, 1]

    # Replaced from ONNX metadata when use_modelmeta_config=True.
    action_scales: list[float] = [1.0] * 25
