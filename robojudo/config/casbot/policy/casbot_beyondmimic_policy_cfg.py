from robojudo.policy.policy_cfgs import BeyondMimicPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class CasbotBeyondMimicDoF(DoFConfig):
    joint_names: list[str] = [
        *[
            'left_leg_pelvic_pitch_joint',
            'right_leg_pelvic_pitch_joint',
            'waist_yaw_joint',
            'left_leg_pelvic_roll_joint',
            'right_leg_pelvic_roll_joint',
            'head_yaw_joint',
            'left_shoulder_pitch_joint',
            'right_shoulder_pitch_joint',
            'left_leg_pelvic_yaw_joint',
            'right_leg_pelvic_yaw_joint',
            'head_pitch_joint',
            'left_shoulder_roll_joint',
            'right_shoulder_roll_joint',
            'left_leg_knee_pitch_joint',
            'right_leg_knee_pitch_joint',
            'left_shoulder_yaw_joint',
            'right_shoulder_yaw_joint',
            'left_leg_ankle_pitch_joint',
            'right_leg_ankle_pitch_joint',
            'left_elbow_pitch_joint',
            'right_elbow_pitch_joint',
            'left_leg_ankle_roll_joint',
            'right_leg_ankle_roll_joint',
            'left_wrist_yaw_joint',
            'right_wrist_yaw_joint',
        ],
    ]

    default_pos: list[float] | None = [
        *[       
            -0.1,   # left_leg_pelvic_pitch_joint
            -0.1,   # right_leg_pelvic_pitch_joint
            0.0,    # waist_yaw_joint
            0.0,    # left_leg_pelvic_roll_joint
            0.0,    # right_leg_pelvic_roll_joint
            0.0,    # head_yaw_joint
            0.0,    # left_shoulder_pitch_joint
            0.0,    # right_shoulder_pitch_joint
            0.0,    # left_leg_pelvic_yaw_joint
            0.0,    # right_leg_pelvic_yaw_joint
            0.0,    # head_pitch_joint
            0.0,    # left_shoulder_roll_joint
            0.0,    # right_shoulder_roll_joint
            0.5,    # left_leg_knee_pitch_joint
            0.5,    # right_leg_knee_pitch_joint
            0.0,    # left_shoulder_yaw_joint
            0.0,    # right_shoulder_yaw_joint
            -0.175, # left_leg_ankle_pitch_joint
            -0.175, # right_leg_ankle_pitch_joint
            -0.5,   # left_elbow_pitch_joint
            -0.5,   # right_elbow_pitch_joint
            0.0,    # left_leg_ankle_roll_joint
            0.0,    # right_leg_ankle_roll_joint
            0.0,    # left_wrist_yaw_joint
            0.0,    # right_wrist_yaw_joint
        ],
    ]

    stiffness: list[float] | None = [
        *[
            276.311, # left_leg_pelvic_pitch_joint
            276.311, # right_leg_pelvic_pitch_joint
            276.311, # waist_yaw_joint
            276.311, # left_leg_pelvic_roll_joint
            276.311, # right_leg_pelvic_roll_joint
            0.0,     # head_yaw_joint
            130.201, # left_shoulder_pitch_joint
            130.201, # right_shoulder_pitch_joint
            156.310, # left_leg_pelvic_yaw_joint
            156.310, # right_leg_pelvic_yaw_joint
            0.0,     # head_pitch_joint
            130.201, # left_shoulder_roll_joint
            130.201, # right_shoulder_roll_joint
            276.311, # left_leg_knee_pitch_joint
            276.311, # right_leg_knee_pitch_joint
            96.825,  # left_shoulder_yaw_joint
            96.825,  # right_shoulder_yaw_joint
            156.310, # left_leg_ankle_pitch_joint
            156.310, # right_leg_ankle_pitch_joint
            130.201, # left_elbow_pitch_joint
            130.201, # right_elbow_pitch_joint
            156.310, # left_leg_ankle_roll_joint
            156.310, # right_leg_ankle_roll_joint
            96.825,  # left_wrist_yaw_joint
            96.825,  # right_wrist_yaw_joint
        ],
    ]

    damping: list[float] | None = [
        *[
            17.590, # left_leg_pelvic_pitch_joint
            17.590, # right_leg_pelvic_pitch_joint
            17.590, # waist_yaw_joint
            17.590, # left_leg_pelvic_roll_joint
            17.590, # right_leg_pelvic_roll_joint
            0.0,    # head_yaw_joint
            8.290,  # left_shoulder_pitch_joint
            8.290,  # right_shoulder_pitch_joint
            9.950,  # left_leg_pelvic_yaw_joint
            9.950,  # right_leg_pelvic_yaw_joint
            0.0,    # head_pitch_joint
            8.290,  # left_shoulder_roll_joint
            8.290,  # right_shoulder_roll_joint
            17.590, # left_leg_knee_pitch_joint
            17.590, # right_leg_knee_pitch_joint
            6.162,  # left_shoulder_yaw_joint
            6.162,  # right_shoulder_yaw_joint
            9.950,  # left_leg_ankle_pitch_joint
            9.950,  # right_leg_ankle_pitch_joint
            8.290,  # left_elbow_pitch_joint
            8.290,  # right_elbow_pitch_joint
            9.950,  # left_leg_ankle_roll_joint
            9.950,  # right_leg_ankle_roll_joint
            6.162,  # left_wrist_yaw_joint
            6.162,  # right_wrist_yaw_joint
        ],
    ]


class CasbotBeyondMimicPolicyCfg(BeyondMimicPolicyCfg):
    robot: str = "casbot"

    # policy_name: str = "Jump_wose"
    policy_name: str = "Dance_wose"
    # policy_name: str = "Violin"
    # policy_name: str = "Waltz"

    obs_dof: DoFConfig = CasbotBeyondMimicDoF()
    action_dof: DoFConfig = obs_dof

    action_beta: float = 1.0
    # ======= POLICY SPECIFIC CONFIGURATION =======
    without_state_estimator: bool = True

    action_scales: list[float] = [
        *[
            0.136, # left_leg_pelvic_pitch_joint
            0.136, # right_leg_pelvic_pitch_joint
            0.054, # waist_yaw_joint
            0.136, # left_leg_pelvic_roll_joint
            0.136, # right_leg_pelvic_roll_joint
            0.0,   # head_yaw_joint
            0.144, # left_shoulder_pitch_joint
            0.144, # right_shoulder_pitch_joint
            0.096, # left_leg_pelvic_yaw_joint
            0.096, # right_leg_pelvic_yaw_joint
            0.0,   # head_pitch_joint
            0.144, # left_shoulder_roll_joint
            0.144, # right_shoulder_roll_joint
            0.136, # left_leg_knee_pitch_joint
            0.136, # right_leg_knee_pitch_joint
            0.093, # left_shoulder_yaw_joint
            0.093, # right_shoulder_yaw_joint
            0.096, # left_leg_ankle_pitch_joint
            0.096, # right_leg_ankle_pitch_joint
            0.144, # left_elbow_pitch_joint
            0.144, # right_elbow_pitch_joint
            0.096, # left_leg_ankle_roll_joint
            0.096, # right_leg_ankle_roll_joint
            0.093, # left_wrist_yaw_joint
            0.093, # right_wrist_yaw_joint
        ],
    ]


class CasbotBeyondMimicResidualPolicyCfg(CasbotBeyondMimicPolicyCfg):
    """
    G1 BeyondMimic policy with residual action mode.
    pd_target = action * scale + motion_joint_pos  (instead of + default_dof_pos)
    The model only outputs residual actions on top of the current reference motion frame.
    """

    use_residual_action: bool = True
