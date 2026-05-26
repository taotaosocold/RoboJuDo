from robojudo.config import ASSETS_DIR
from robojudo.environment.env_cfgs import EnvCfg
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg


class Casbot_25DoF(DoFConfig):
    # num_dofs as 25
    joint_names: list[str] = [
        *[
            'left_leg_pelvic_pitch_joint', 'left_leg_pelvic_roll_joint', 'left_leg_pelvic_yaw_joint', 'left_leg_knee_pitch_joint', 'left_leg_ankle_pitch_joint', 'left_leg_ankle_roll_joint',
            'right_leg_pelvic_pitch_joint', 'right_leg_pelvic_roll_joint', 'right_leg_pelvic_yaw_joint', 'right_leg_knee_pitch_joint', 'right_leg_ankle_pitch_joint', 'right_leg_ankle_roll_joint',
            'waist_yaw_joint', 'head_yaw_joint', 'head_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_pitch_joint', 'right_wrist_yaw_joint'
        ],
    ]
    default_pos: list[float] | None = [
        *[
            -0.1, 0.0, 0.0, 0.5, -0.175, 0.0,
            -0.1, 0.0, 0.0, 0.5, -0.175, 0.0,
            0.0, 0.0,  0.0,
            0.0,  0.0, 0.0, -0.5, 0.0,
            0.0,  0.0, 0.0, -0.5, 0.0,
        ],
    ]

    stiffness: list[float] | None = [
        *[
            276.311, 276.311, 156.310, 276.311, 156.310, 156.310,  # left leg:  pelvic_pitch, pelvic_roll, pelvic_yaw, knee_pitch, ankle_pitch, ankle_roll
            276.311, 276.311, 156.310, 276.311, 156.310, 156.310,  # right leg: pelvic_pitch, pelvic_roll, pelvic_yaw, knee_pitch, ankle_pitch, ankle_roll
            276.311, 0.000,   0.000,                              # waist_yaw, head_yaw, head_pitch (head: no actuator in training → 0)
            130.201, 130.201,  96.825, 130.201,  96.825,           # left arm:  sh_pitch, sh_roll, sh_yaw, elbow, wrist_yaw
            130.201, 130.201,  96.825, 130.201,  96.825,
        ],
    ]

    damping: list[float] | None = [
        *[
            17.591, 17.591,  9.951, 17.591,  9.951,  9.951,
            17.591, 17.591,  9.951, 17.591,  9.951,  9.951,
            17.591,  0.000,  0.000,
            8.289,  8.289,  6.164,  8.289,  6.164,
            8.289,  8.289,  6.164,  8.289,  6.164,
        ],
    ]

    torque_limits: list[float] | None = [
        *[
            150.0, 150.0,  60.0, 150.0,  60.0,  60.0,
            150.0, 150.0,  60.0, 150.0,  60.0,  60.0,
            60.0,   0.0,   0.0,
            75.0,  75.0,  36.0,  75.0,  36.0,
            75.0,  75.0,  36.0,  75.0,  36.0,
        ],
    ]

    position_limits: list[list[float]] | None = [
        *[
            [-1.9199, 1.5708],   # left_leg_pelvic_pitch_joint
            [-0.17453, 1.5708],  # left_leg_pelvic_roll_joint
            [-1.5708, 1.5708],   # left_leg_pelvic_yaw_joint
            [0.0, 2.5307],       # left_leg_knee_pitch_joint
            [-0.87266, 0.50614], # left_leg_ankle_pitch_joint
            [-0.50614, 0.50614], # left_leg_ankle_roll_joint
            [-1.9199, 1.5708],   # right_leg_pelvic_pitch_joint
            [-1.5708, 0.17453],  # right_leg_pelvic_roll_joint
            [-1.5708, 1.5708],   # right_leg_pelvic_yaw_joint
            [0.0, 2.5307],       # right_leg_knee_pitch_joint
            [-0.87266, 0.50614], # right_leg_ankle_pitch_joint
            [-0.50614, 0.50614], # right_leg_ankle_roll_joint
            [-1.5708, 1.5708],   # waist_yaw_joint
            [-1.5708, 1.5708],   # head_yaw_joint
            [-0.2618, 0.5236],   # head_pitch_joint
            [-3.1416, 1.0472],   # left_shoulder_pitch_joint
            [-0.3491, 3.1416],   # left_shoulder_roll_joint
            [-1.5708, 1.5708],   # left_shoulder_yaw_joint
            [-1.8675, 0.0],      # left_elbow_pitch_joint
            [-1.5708, 1.5708],   # left_wrist_yaw_joint
            [-3.1416, 1.0472],   # right_shoulder_pitch_joint
            [-3.1416, 0.3491],   # right_shoulder_roll_joint
            [-1.5708, 1.5708],   # right_shoulder_yaw_joint
            [-1.8675, 0.0],      # right_elbow_pitch_joint
            [-1.5708, 1.5708],   # right_wrist_yaw_joint
        ],
    ]


class CasbotEnvCfg(EnvCfg):
    xml: str = (ASSETS_DIR / "robots/casbot_skeleton/casbot_skeleton_25dof.xml").as_posix()

    dof: DoFConfig = Casbot_25DoF()

    forward_kinematic: ForwardKinematicCfg | None = ForwardKinematicCfg(
        xml_path=xml,
        debug_viz=False,
        kinematic_joint_names=dof.joint_names,
    )
    update_with_fk: bool = True
    torso_name: str = "waist_yaw_link"
