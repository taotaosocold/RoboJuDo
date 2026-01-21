from robojudo.controller.ctrl_cfgs import BeyondMimicCtrlCfg, BeyondMimicMotionTrackingCtrlCfg


class G1BeyondmimicCtrlCfg(BeyondMimicCtrlCfg):
    # ==== motion config ====
    robot: str = "g1"  # default as only supported g1
    motion_name: str = "dance1_subject2"

    motion_cfg: BeyondMimicCtrlCfg.MotionCommandCfg = BeyondMimicCtrlCfg.MotionCommandCfg(
        anchor_body_name="torso_link",
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
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        body_names_all=[
            "pelvis",
            "left_hip_pitch_link",
            "right_hip_pitch_link",
            "waist_yaw_link",
            "left_hip_roll_link",
            "right_hip_roll_link",
            "waist_roll_link",
            "left_hip_yaw_link",
            "right_hip_yaw_link",
            "torso_link",
            "left_knee_link",
            "right_knee_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_ankle_pitch_link",
            "right_ankle_pitch_link",
            "left_shoulder_roll_link",
            "right_shoulder_roll_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link",
            "left_wrist_pitch_link",
            "right_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
    )

class G1BeyondmimicCtrlCfg23DOF(BeyondMimicCtrlCfg):
    # ==== motion config ====
    robot: str = "g1"  # default as only supported g1
    motion_name: str = "dance1_subject2"

    motion_cfg: BeyondMimicCtrlCfg.MotionCommandCfg = BeyondMimicCtrlCfg.MotionCommandCfg(
        anchor_body_name="torso_link",
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
        ],
    )

class G1BeyondmimicMotionTrackingCtrlCfg(BeyondMimicMotionTrackingCtrlCfg):
    # ==== motion config ====
    robot: str = "g1"  # default as only supported g1
    motion_name: str = "dance1_subject2"

    motion_cfg: BeyondMimicCtrlCfg.MotionCommandCfg = BeyondMimicMotionTrackingCtrlCfg.MotionCommandCfg(
        anchor_body_name="torso_link",
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
        ],
    )