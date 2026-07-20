from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
)
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.casbot_beyondmimic_ctrl_cfg import CasbotBeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import (  # noqa: F401
    G1MotionCtrlCfg,
    G1MotionH2HCtrlCfg,
    G1MotionKungfuBotCtrlCfg,
    G1MotionTwistCtrlCfg,
)
from .ctrl.g1_twist_redis_ctrl_cfg import G1TwistRedisCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.casbot_mujuco_env_cfg import CasbotMujocoEnvCfg  # noqa: F401
from .env.casbot_real_env_cfg import CasbotHlCfg, CasbotRealEnvCfg  # noqa: F401
from .env.g1_mujuco_env_cfg import G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg  # noqa: F401
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg, G1BeyondMimicResidualPolicyCfg  # noqa: F401
from .policy.casbot_beyondmimic_policy_cfg import CasbotBeyondMimicPolicyCfg, CasbotBeyondMimicResidualPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401


# ======================== Basic Configs ======================== #
@cfg_registry.register
class g1(RlPipelineCfg):
    """
    Unitree G1 robot configuration, Unitree Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    # env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()
    # env: G1_12MujocoEnvCfg = G1_12MujocoEnvCfg()

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        JoystickCtrlCfg(),
        # KeyboardCtrlCfg(),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    # policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    # policy: G1AmoPolicyCfg = G1AmoPolicyCfg()

    # run_fullspeed: bool = env.is_sim


@cfg_registry.register
class g1_real(g1):
    """
    Unitree G1 robot, Unitree Policy, Sim2Real.
    To extend the sim2sim config to sim2real, just need to change the env to real env.
    """

    # env: G1DummyEnvCfg = G1DummyEnvCfg()
    env: G1RealEnvCfg = G1RealEnvCfg(
        # env_type="UnitreeEnv",  # For unitree_sdk2py
        env_type="UnitreeCppEnv",  # For unitree_cpp, check README for more details
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


@cfg_registry.register
class g1_switch(RlMultiPolicyPipelineCfg):
    """
    Example of multi-policy pipeline configuration.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        # KeyboardCtrlCfg(
        #     triggers_extra={
        #         "Key.tab": "[POLICY_TOGGLE]",
        #     }
        # ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Up": "[POLICY_SWITCH],1",
            }
        ),
    ]

    policies: list[G1UnitreePolicyCfg | G1AmoPolicyCfg] = [
        G1UnitreePolicyCfg(),
        G1AmoPolicyCfg(),
    ]


@cfg_registry.register
class g1_locomimic(RlLocoMimicPipelineCfg):
    """
    Example of loco mimic pipeline configuration.
    You can switch between loco and mimic policies during runtime, with interpolation.
    === Check more fancy locomimic examples in g1_loco_mimic_cfg.py ===
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_LOCO]",
                "RB+Up": "[POLICY_MIMIC]",
            }
        ),
    ]

    loco_policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    mimic_policies: list[G1AsapPolicyCfg] = [
        G1AsapPolicyCfg(),
    ]


# ======================== Configs for supported Policy ======================== #


@cfg_registry.register
class g1_h2h(RlPipelineCfg):
    """
    Human2Humanoid
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionH2HCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionH2HCtrlCfg(),
    ]

    policy: G1H2HPolicyCfg = G1H2HPolicyCfg()


@cfg_registry.register
class casbot_beyondmimic(RlPipelineCfg):
    """
    Casbot BeyondMimic Policy, motion embedded in onnx (use_motion_from_model=True).
    """

    robot: str = "casbot"
    env: CasbotMujocoEnvCfg = CasbotMujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "1": "[STATE_PASSIVE]",
                "2": "[STATE_FIXED_STAND]",
                "3": "[STATE_POLICY]",
                "0": "[STATE_ESTOP]",
            },
        ),
    ]

    policy: CasbotBeyondMimicPolicyCfg = CasbotBeyondMimicPolicyCfg(
        policy_name="fk_beyondmimic",
        without_state_estimator=True,
        use_modelmeta_config=True,
        use_motion_from_model=True,
        max_timestep=300,
    )

    fsm_enabled: bool = True
    do_safety_check: bool = True
    stand_target_pos: list[float] = [  # FK model first frame (fk), Casbot_25DoF order
        # left leg
        -0.1010930389, 0.1863300502, 0.1534859538, 0.1218007132, -0.0273471251, 0.0,
        # right leg
        -0.1159527600, -0.1466802061, -0.1086166725, 0.1400099695, 0.0167938694, 0.0,
        # waist_yaw, head_yaw, head_pitch
        -0.0127099706, -0.0008552494, 0.0,
        # left arm
        -0.0597212315, 0.7225397229, -0.0980680510, -0.2135050893, -0.0890462548,
        # right arm
        -0.0829682425, -0.8226424456, 0.0251893010, -0.1944101751, 0.1169776917,
    ]

# export PYTHONPATH=/home/casbot/Desktop/RoboJuDo:$PYTHONPATH
@cfg_registry.register
class casbot_beyondmimic_real(RlPipelineCfg):
    """
    Casbot BeyondMimic Policy on real robot via hl_motion ROS2 SDK.
    Run with: python scripts/run_pipeline.py -c casbot_beyondmimic_real
    """

    robot: str = "casbot"
    env: CasbotRealEnvCfg = CasbotRealEnvCfg(
        hl=CasbotHlCfg(
            robot_joint_names=[
                # left leg (hl_motion naming)
                "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
                # right leg
                "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
                # head, waist (must match WBC FSM order: head_yaw, head_pitch, waist_yaw)
                "head_yaw_joint", "head_pitch_joint", "waist_yaw_joint",
                # left arm
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_wrist_yaw_joint",
                # right arm
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_wrist_yaw_joint",
            ],
        ),
    )
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "Key.esc": "[SHUTDOWN]",
                "1": "[STATE_PASSIVE]",
                "2": "[STATE_FIXED_STAND]",
                "3": "[STATE_POLICY]",
                "0": "[STATE_ESTOP]",
            },
        ),
    ]

    policy: CasbotBeyondMimicPolicyCfg = CasbotBeyondMimicPolicyCfg(
        policy_name="fk_beyondmimic",
        without_state_estimator=True,
        use_modelmeta_config=True,
        use_motion_from_model=True,
        max_timestep=300,
    )

    fsm_enabled: bool = True
    do_safety_check: bool = True
    stand_target_pos: list[float] = [  # FK model first frame (fk), Casbot_25DoF order
        # left leg
        -0.1010930389, 0.1863300502, 0.1534859538, 0.1218007132, -0.0273471251, 0.0,
        # right leg
        -0.1159527600, -0.1466802061, -0.1086166725, 0.1400099695, 0.0167938694, 0.0,
        # waist_yaw, head_yaw, head_pitch
        -0.0127099706, -0.0008552494, 0.0,
        # left arm
        -0.0597212315, 0.7225397229, -0.0980680510, -0.2135050893, -0.0890462548,
        # right arm
        -0.0829682425, -0.8226424456, 0.0251893010, -0.1944101751, 0.1169776917,
    ]


@cfg_registry.register
class casbot_beyondmimic_with_ctrl(RlPipelineCfg):
    """
    Casbot BeyondMimic Policy, motion from external CasbotBeyondmimicCtrlCfg (use_motion_from_model=False).
    Put motion .npz files in assets/motions/casbot/beyondmimic/.
    """

    robot: str = "casbot"
    env: CasbotMujocoEnvCfg = CasbotMujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | CasbotBeyondmimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        CasbotBeyondmimicCtrlCfg(
            motion_name="118_01_poses",
        ),
    ]

    policy: CasbotBeyondMimicPolicyCfg = CasbotBeyondMimicPolicyCfg(
        policy_name="policy",
        without_state_estimator=True,
        use_modelmeta_config=True,
        use_motion_from_model=False,
        max_timestep=3000,
    )


@cfg_registry.register
class casbot_beyondmimic_residual(RlPipelineCfg):
    """
    Casbot BeyondMimic residual action policy, motion embedded in onnx.
    pd_target = action * scale + motion_joint_pos
    """

    robot: str = "casbot"
    env: CasbotMujocoEnvCfg = CasbotMujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: CasbotBeyondMimicResidualPolicyCfg = CasbotBeyondMimicResidualPolicyCfg(
        policy_name="fallAndGetUp2_subject2_clip_50hz_residual",
        use_modelmeta_config=True,
        use_motion_from_model=True,
        without_state_estimator=True,
        max_timestep=5000,
    )


@cfg_registry.register
class casbot_beyondmimic_residual_with_ctrl(RlPipelineCfg):
    """
    Casbot BeyondMimic residual action policy, motion from external CasbotBeyondmimicCtrlCfg.
    pd_target = action * scale + motion_joint_pos
    Put motion .npz files in assets/motions/casbot/beyondmimic/.
    """

    robot: str = "casbot"
    env: CasbotMujocoEnvCfg = CasbotMujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | CasbotBeyondmimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        CasbotBeyondmimicCtrlCfg(
            motion_name="dance1_subject2",
        ),
    ]

    policy: CasbotBeyondMimicResidualPolicyCfg = CasbotBeyondMimicResidualPolicyCfg(
        policy_name="dance1_subject1_residual",
        use_modelmeta_config=True,
        use_motion_from_model=False,
        without_state_estimator=True,
    )

@cfg_registry.register
class g1_beyondmimic_with_ctrl(RlPipelineCfg):
    """
    BeyondMimic with External BeyondMimicCtrl as motion source.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1BeyondmimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1BeyondmimicCtrlCfg(
            motion_name="dance1_subject1",  # you can put your own motion file in assets/motions/g1
        ),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="dance1_subject1_robotlab",
        use_modelmeta_config=False,
        use_motion_from_model=False,  # use motion from BeyondmimicCtrl instead of the onnx
    )


@cfg_registry.register
class g1_beyondmimic_residual(RlPipelineCfg):
    """
    G1 BeyondMimic residual action policy.
    The model outputs residual actions: pd_target = action * scale + motion_joint_pos
    Requires an external BeyondMimicCtrl to provide the reference motion sequence.
    Run with: python scripts/run_pipeline.py -c g1_beyondmimic_residual
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1BeyondmimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1BeyondmimicCtrlCfg(
            motion_name="G1_Take_102.bvh_60hz",  # change to match your motion file
        ),
    ]

    policy: G1BeyondMimicResidualPolicyCfg = G1BeyondMimicResidualPolicyCfg(
        policy_name="policy",  # change to your model name (onnx filename without extension)
        use_modelmeta_config=False,
        use_motion_from_model=False,  # motion comes from BeyondMimicCtrl, not the onnx
        without_state_estimator=True,
    )


@cfg_registry.register
class g1_asap(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Policy, Sim2Sim.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(triggers={"i": "[SIM_REBORN]", "o": "[SHUTDOWN]", "r": "[MOTION_RESET]"}),
    ]

    policy: G1AsapPolicyCfg = G1AsapPolicyCfg()
    """You can also try other models, from ASAP, RoboMimic, KungfuBot(PBHC)"""
    # policy: G1KungfuBotPolicyCfg = G1KungfuBotPolicyCfg() # KungfuBot horse_squat
    # # fmt: off
    # policy: G1AsapPolicyCfg = G1AsapPolicyCfg(
    #     policy_name="robomimic",
    #     relative_path="dance_0605.onnx",
    #     motion_length_s=18.0,
    #     start_upper_body_dof_pos = [
    #         0, 0, 0,
    #         0.35, 0.18, 0, 0.87,
    #         0.35, -0.18, 0, 0.87,
    #     ],
    # )
    # # fmt: on


@cfg_registry.register
class g1_asap_loco(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Locomotion Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
            }
        ),
    ]

    policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()


@cfg_registry.register
class g1_kungfubot2(RlPipelineCfg):
    """
    PBHC KungfuBot2 General Policy
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionKungfuBotCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionKungfuBotCtrlCfg(
            motion_name="kungfubot/Horse-stance_pose",  # put motion files in assets/motions/g1/phc/kungfubot
        ),
    ]

    policy: G1KungfuBotGeneralPolicyCfg = G1KungfuBotGeneralPolicyCfg(
        policy_name="horse_test_43000",  # this is a test model trained with only one motion
        compatibility_old_version=True,  # for old version of kungfubot general policy (before 2025-11-13 bugfix #68)
    )


@cfg_registry.register
class g1_twist(RlPipelineCfg):
    """
    Unitree G1 robot configuration, TWIST Policy, Sim2Sim.
    TwistRedisCtrl for the original repo of high level motion stream over redis.
    MotionTwistCtrl for built-in motion control.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[G1TwistRedisCtrlCfg | G1MotionTwistCtrlCfg] = [  # note: the ranking of controllers matters
        G1TwistRedisCtrlCfg(redis_host="localhost"),  # with hign level motion lib through redis
        # G1MotionTwistCtrlCfg(), # with built-in motion ctrl
    ]

    policy: G1TwistPolicyCfg = G1TwistPolicyCfg()


# ======================== Fancy Example Configs ======================== #


@cfg_registry.register
class g1_switch_beyondmimic(RlMultiPolicyPipelineCfg):
    """
    Switch between multiple BeyondMimic policies. Withour Interpolation.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                "!": "[POLICY_SWITCH],0",  # note: with shift
                "@": "[POLICY_SWITCH],1",  # note: with shift
                "#": "[POLICY_SWITCH],2",  # note: with shift
                "$": "[POLICY_SWITCH],3",  # note: with shift
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Left": "[POLICY_SWITCH],1",
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
            }
        ),
    ]

    policies: list[G1AmoPolicyCfg | G1BeyondMimicPolicyCfg] = [
        G1AmoPolicyCfg(),
        G1BeyondMimicPolicyCfg(policy_name="Violin", without_state_estimator=False, max_timestep=500),
        G1BeyondMimicPolicyCfg(policy_name="Waltz", without_state_estimator=False, max_timestep=850),
        G1BeyondMimicPolicyCfg(policy_name="Dance_wose", without_state_estimator=True),
    ]


# TIPS: check g1_loco_mimic_cfg.py for more complex examples
