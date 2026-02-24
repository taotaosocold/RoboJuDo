from robojudo.policy.policy_cfgs import VaeMimicPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

class G1Dof(DoFConfig):
    joint_names: list[str] = [
        *['left_hip_pitch_joint', 
          'right_hip_pitch_joint', 
          'waist_yaw_joint', 
          'left_hip_roll_joint', 
          'right_hip_roll_joint', 
          'left_shoulder_pitch_joint', 
          'right_shoulder_pitch_joint', 
          'left_hip_yaw_joint', 
          'right_hip_yaw_joint', 
          'left_shoulder_roll_joint', 
          'right_shoulder_roll_joint', 
          'left_knee_joint', 
          'right_knee_joint', 
          'left_shoulder_yaw_joint', 
          'right_shoulder_yaw_joint', 
          'left_ankle_pitch_joint', 
          'right_ankle_pitch_joint', 
          'left_elbow_joint', 
          'right_elbow_joint', 
          'left_ankle_roll_joint', 
          'right_ankle_roll_joint', 
          'left_wrist_roll_joint', 
          'right_wrist_roll_joint'],
    ]

    default_pos: list[float] | None = [
        *[-0.302, -0.319,  0.001,  0.,     0.005,  0.202,  0.209,  0.008,  0.005,  0.203,
        -0.199,  0.669,  0.671, -0.005,  0.008, -0.369, -0.359,  0.594,  0.61,   0.003,
        -0.,     0.005, -0.004]
    ]

    stiffness: list[float] | None = [
        *[40.179, 40.179, 40.179, 99.098, 99.098, 14.251, 14.251, 40.179, 40.179, 14.251,
        14.251, 99.098, 99.098, 14.251, 14.251, 28.501, 28.501, 14.251, 14.251, 28.501,
        28.501, 14.251, 14.251]
    ]

    damping: list[float] | None = [
        *[2.558, 2.558, 2.558, 6.309, 6.309, 0.907, 0.907, 2.558, 2.558, 0.907, 0.907, 6.309,
        6.309, 0.907, 0.907, 1.814, 1.814, 0.907, 0.907, 1.814, 1.814, 0.907, 0.907]
    ]


class G1VaeMimicPolicyCfg(VaeMimicPolicyCfg):
    robot: str = "g1"
    policy_name: str = "0212"
    obs_dof: DoFConfig = G1Dof()
    action_dof: DoFConfig = obs_dof

    action_beta: float = 1.0
    without_state_estimator: bool = True

    action_scales: list[float] = [
        0.548, 0.548, 0.548, 0.351, 0.351, 0.439, 0.439, 0.548, 0.548, 0.439, 0.439, 0.351,
        0.351, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439
    ]