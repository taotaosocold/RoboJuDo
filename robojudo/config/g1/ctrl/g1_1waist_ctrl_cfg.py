
from robojudo.controller.ctrl_cfgs import VaeMimicCtrlCfg

class G1VaeMimicCtrlCfg(VaeMimicCtrlCfg):
    robot: str = "g1"
    motion_name: str = '90_08_poses'
    code_path: str = "/home/ubuntu/projects/hjj-robot_lab/source/vq-vae/logs/2026-02-18_14-06-53/codebook.npz"
    code_key: str = "/home/ubuntu/projects/hjj-robot_lab/source/motion/motion_hjq/dance/90_08_poses.npz"

    motion_cfg: VaeMimicCtrlCfg.MotionCommandCfg = VaeMimicCtrlCfg.MotionCommandCfg(
        anchor_body_name="pelvis",
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_roll_rubber_hand",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_roll_rubber_hand",
        ],
        body_names_all=[
            'pelvis', 
            'left_hip_pitch_link', 
            'pelvis_contour_link', 
            'right_hip_pitch_link', 
            'torso_link', 
            'left_hip_roll_link', 
            'right_hip_roll_link', 
            'head_link', 
            'left_shoulder_pitch_link', 
            'logo_link', 
            'right_shoulder_pitch_link', 
            'left_hip_yaw_link', 
            'right_hip_yaw_link', 
            'left_shoulder_roll_link', 
            'right_shoulder_roll_link', 
            'left_knee_link', 
            'right_knee_link', 
            'left_shoulder_yaw_link', 
            'right_shoulder_yaw_link', 
            'left_ankle_pitch_link', 
            'right_ankle_pitch_link', 
            'left_elbow_link', 
            'right_elbow_link', 
            'left_ankle_roll_link', 
            'right_ankle_roll_link', 
            'left_wrist_roll_rubber_hand', 
            'right_wrist_roll_rubber_hand'
            ]
    )