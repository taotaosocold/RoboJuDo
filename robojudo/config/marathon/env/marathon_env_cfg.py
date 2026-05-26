from robojudo.config import ASSETS_DIR
from robojudo.environment.env_cfgs import EnvCfg
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg


class Marathon_18DoF(DoFConfig):
    # num_dofs as 18
    joint_names: list[str] = [
        *[
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_elbow_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_elbow_pitch_joint',
            'right_leg_pelvic_pitch_joint', 'right_leg_pelvic_roll_joint', 'right_leg_pelvic_yaw_joint', 'right_leg_knee_pitch_joint', 'right_leg_ankle_pitch_joint', 'right_leg_ankle_roll_joint',
            'left_leg_pelvic_pitch_joint', 'left_leg_pelvic_roll_joint', 'left_leg_pelvic_yaw_joint', 'left_leg_knee_pitch_joint', 'left_leg_ankle_pitch_joint', 'left_leg_ankle_roll_joint'
        ],
    ]
    default_pos: list[float] | None = [
        *[
            0.0, 0.0, -0.5,   # right arm: sh_pitch, sh_roll, elbow
            0.0, 0.0, -0.5,   # left arm:  sh_pitch, sh_roll, elbow
            -0.1, 0.0, 0.0, 0.5, -0.175, 0.0,  # right leg: pelvic_pitch, pelvic_roll, pelvic_yaw, knee, ankle_pitch, ankle_roll
            -0.1, 0.0, 0.0, 0.5, -0.175, 0.0,  # left leg:  pelvic_pitch, pelvic_roll, pelvic_yaw, knee, ankle_pitch, ankle_roll
        ],
    ]

    stiffness: list[float] | None = [
        *[
            130.199, 130.199, 130.199,   # right arm: sh_pitch, sh_roll, elbow
            130.199, 130.199, 130.199,   # left arm:  sh_pitch, sh_roll, elbow
            785.511, 1094.436, 241.440, 785.511, 130.199, 130.199,  # right leg: pelvic_pitch, pelvic_roll, pelvic_yaw, knee, ankle_pitch, ankle_roll
            785.511, 1094.436, 241.440, 785.511, 130.199, 130.199,  # left leg:  pelvic_pitch, pelvic_roll, pelvic_yaw, knee, ankle_pitch, ankle_roll
        ],
    ]

    damping: list[float] | None = [
        *[
            8.286, 8.286, 8.286,   # right arm
            8.286, 8.286, 8.286,   # left arm
            50.007, 69.676, 15.370, 50.007, 8.286, 8.286,  # right leg
            50.007, 69.676, 15.370, 50.007, 8.286, 8.286,  # left leg
        ],
    ]

    torque_limits: list[float] | None = [
        *[
            72.0, 72.0, 72.0,   # right arm
            72.0, 72.0, 72.0,   # left arm
            255.0, 212.0, 110.0, 212.0, 72.0, 72.0,  # right leg
            255.0, 212.0, 110.0, 212.0, 72.0, 72.0,  # left leg
        ],
    ]

    position_limits: list[list[float]] | None = [
        *[
            [-3.1416, 1.0472],   # right_shoulder_pitch_joint
            [-3.1416, 0.3491],   # right_shoulder_roll_joint
            [-1.8675, 0.0],      # right_elbow_pitch_joint
            [-3.1416, 1.0472],   # left_shoulder_pitch_joint
            [-0.3491, 3.1416],   # left_shoulder_roll_joint
            [-1.8675, 0.0],      # left_elbow_pitch_joint
            [-1.9199, 1.5708],   # right_leg_pelvic_pitch_joint
            [-1.5708, 0.17453],  # right_leg_pelvic_roll_joint
            [-1.5708, 1.5708],   # right_leg_pelvic_yaw_joint
            [0.0, 2.5307],       # right_leg_knee_pitch_joint
            [-0.87266, 0.50614], # right_leg_ankle_pitch_joint
            [-0.50614, 0.50614], # right_leg_ankle_roll_joint
            [-1.9199, 1.5708],   # left_leg_pelvic_pitch_joint
            [-0.17453, 1.5708],  # left_leg_pelvic_roll_joint
            [-1.5708, 1.5708],   # left_leg_pelvic_yaw_joint
            [0.0, 2.5307],       # left_leg_knee_pitch_joint
            [-0.87266, 0.50614], # left_leg_ankle_pitch_joint
            [-0.50614, 0.50614], # left_leg_ankle_roll_joint
        ],
    ]


class MarathonEnvCfg(EnvCfg):
    xml: str = (ASSETS_DIR / "robots/marathon/marathon_001.xml").as_posix()

    dof: DoFConfig = Marathon_18DoF()

    forward_kinematic: ForwardKinematicCfg | None = ForwardKinematicCfg(
        xml_path=xml,
        debug_viz=False,
        kinematic_joint_names=dof.joint_names,
    )
    update_with_fk: bool = True
    torso_name: str = "base_link"