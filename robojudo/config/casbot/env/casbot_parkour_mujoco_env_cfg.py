from robojudo.config import ASSETS_DIR

from .casbot_mujuco_env_cfg import CasbotMujocoEnvCfg


class CasbotParkourMujocoEnvCfg(CasbotMujocoEnvCfg):
    """MuJoCo environment for one aligned CASBOT motion/terrain pair."""

    xml: str = (
        ASSETS_DIR / "robots/casbot_skeleton/casbot_skeleton_25dof_rev_1_0_scene2.xml"
    ).as_posix()
    use_height_scan: bool = True
    height_scan_z_clip: list[float] = [-20.0, 20.0]
    height_scan_body: str = "waist_yaw_link"
    # The scene XML contains both the visual STL and its box collision geometry.
    terrain_stl: str | None = None
