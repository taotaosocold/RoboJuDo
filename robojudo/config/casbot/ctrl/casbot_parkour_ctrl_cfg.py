from robojudo.config import ASSETS_DIR

from .casbot_beyondmimic_ctrl_cfg import CasbotBeyondmimicCtrlCfg


class CasbotParkourCtrlCfg(CasbotBeyondmimicCtrlCfg):
    """Single reference motion paired with the CASBOT Parkour policy."""

    motion_name: str = "walking"
    override_robot_anchor_pos: bool = False

    @property
    def motion_path(self) -> str:
        return (ASSETS_DIR / f"motions/casbot/parkour/{self.motion_name}.npz").as_posix()
