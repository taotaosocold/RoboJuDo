from robojudo.environment.env_cfgs import MujocoEnvCfg

from .casbot_env_cfg import CasbotEnvCfg


class CasbotMujocoEnvCfg(CasbotEnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======

    update_with_fk: bool = True

