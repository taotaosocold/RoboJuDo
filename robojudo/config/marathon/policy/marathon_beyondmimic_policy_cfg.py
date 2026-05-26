from robojudo.policy.policy_cfgs import BeyondMimicPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class MarathonBeyondMimicDoF(DoFConfig):
    joint_names: list[str] = [
        'left_leg_pelvic_pitch_joint',
        'left_shoulder_pitch_joint',
        'right_leg_pelvic_pitch_joint',
        'right_shoulder_pitch_joint',
        'left_leg_pelvic_roll_joint',
        'left_shoulder_roll_joint',
        'right_leg_pelvic_roll_joint',
        'right_shoulder_roll_joint',
        'left_leg_pelvic_yaw_joint',
        'left_elbow_pitch_joint',
        'right_leg_pelvic_yaw_joint',
        'right_elbow_pitch_joint',
        'left_leg_knee_pitch_joint',
        'right_leg_knee_pitch_joint',
        'left_leg_ankle_pitch_joint',
        'right_leg_ankle_pitch_joint',
        'left_leg_ankle_roll_joint',
        'right_leg_ankle_roll_joint',
    ]

    default_pos: list[float] | None = [
        -0.1,   # left_leg_pelvic_pitch_joint
         0.0,   # left_shoulder_pitch_joint
        -0.1,   # right_leg_pelvic_pitch_joint
         0.0,   # right_shoulder_pitch_joint
         0.0,   # left_leg_pelvic_roll_joint
         0.0,   # left_shoulder_roll_joint
         0.0,   # right_leg_pelvic_roll_joint
         0.0,   # right_shoulder_roll_joint
         0.0,   # left_leg_pelvic_yaw_joint
        -0.5,   # left_elbow_pitch_joint
         0.0,   # right_leg_pelvic_yaw_joint
        -0.5,   # right_elbow_pitch_joint
         0.5,   # left_leg_knee_pitch_joint
         0.5,   # right_leg_knee_pitch_joint
        -0.175, # left_leg_ankle_pitch_joint
        -0.175, # right_leg_ankle_pitch_joint
         0.0,   # left_leg_ankle_roll_joint
         0.0,   # right_leg_ankle_roll_joint
    ]

    stiffness: list[float] | None = [
        785.511, # left_leg_pelvic_pitch_joint
        130.199, # left_shoulder_pitch_joint
        785.511, # right_leg_pelvic_pitch_joint
        130.199, # right_shoulder_pitch_joint
        1094.436,# left_leg_pelvic_roll_joint
        130.199, # left_shoulder_roll_joint
        1094.436,# right_leg_pelvic_roll_joint
        130.199, # right_shoulder_roll_joint
        241.440, # left_leg_pelvic_yaw_joint
        130.199, # left_elbow_pitch_joint
        241.440, # right_leg_pelvic_yaw_joint
        130.199, # right_elbow_pitch_joint
        785.511, # left_leg_knee_pitch_joint
        785.511, # right_leg_knee_pitch_joint
        130.199, # left_leg_ankle_pitch_joint
        130.199, # right_leg_ankle_pitch_joint
        130.199, # left_leg_ankle_roll_joint
        130.199, # right_leg_ankle_roll_joint
    ]

    damping: list[float] | None = [
        50.007, # left_leg_pelvic_pitch_joint
        8.286,  # left_shoulder_pitch_joint
        50.007, # right_leg_pelvic_pitch_joint
        8.286,  # right_shoulder_pitch_joint
        69.676, # left_leg_pelvic_roll_joint
        8.286,  # left_shoulder_roll_joint
        69.676, # right_leg_pelvic_roll_joint
        8.286,  # right_shoulder_roll_joint
        15.370, # left_leg_pelvic_yaw_joint
        8.286,  # left_elbow_pitch_joint
        15.370, # right_leg_pelvic_yaw_joint
        8.286,  # right_elbow_pitch_joint
        50.007, # left_leg_knee_pitch_joint
        50.007, # right_leg_knee_pitch_joint
        8.286,  # left_leg_ankle_pitch_joint
        8.286,  # right_leg_ankle_pitch_joint
        8.286,  # left_leg_ankle_roll_joint
        8.286,  # right_leg_ankle_roll_joint
    ]


class MarathonBeyondMimicPolicyCfg(BeyondMimicPolicyCfg):
    robot: str = "marathon"

    # policy_name: str = "Jump_wose"
    policy_name: str = "Dance_wose"
    # policy_name: str = "Violin"
    # policy_name: str = "Waltz"

    obs_dof: DoFConfig = MarathonBeyondMimicDoF()
    action_dof: DoFConfig = obs_dof

    action_beta: float = 1.0
    # ======= POLICY SPECIFIC CONFIGURATION =======
    without_state_estimator: bool = True

    action_scales: list[float] = [
        0.081, # left_leg_pelvic_pitch_joint
        0.138, # left_shoulder_pitch_joint
        0.081, # right_leg_pelvic_pitch_joint
        0.138, # right_shoulder_pitch_joint
        0.048, # left_leg_pelvic_roll_joint
        0.138, # left_shoulder_roll_joint
        0.048, # right_leg_pelvic_roll_joint
        0.138, # right_shoulder_roll_joint
        0.114, # left_leg_pelvic_yaw_joint
        0.138, # left_elbow_pitch_joint
        0.114, # right_leg_pelvic_yaw_joint
        0.138, # right_elbow_pitch_joint
        0.067, # left_leg_knee_pitch_joint
        0.067, # right_leg_knee_pitch_joint
        0.138, # left_leg_ankle_pitch_joint
        0.138, # right_leg_ankle_pitch_joint
        0.138, # left_leg_ankle_roll_joint
        0.138, # right_leg_ankle_roll_joint
    ]


class MarathonBeyondMimicResidualPolicyCfg(MarathonBeyondMimicPolicyCfg):
    """
    Marathon BeyondMimic policy with residual action mode.
    pd_target = action * scale + motion_joint_pos  (instead of + default_dof_pos)
    The model only outputs residual actions on top of the current reference motion frame.
    """
    use_residual_action: bool = True