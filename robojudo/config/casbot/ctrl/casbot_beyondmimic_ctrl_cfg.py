from robojudo.controller.ctrl_cfgs import BeyondMimicCtrlCfg


class CasbotBeyondmimicCtrlCfg(BeyondMimicCtrlCfg):
    robot: str = "casbot"
    motion_name: str = "dance1_subject2"

    motion_cfg: BeyondMimicCtrlCfg.MotionCommandCfg = BeyondMimicCtrlCfg.MotionCommandCfg(
        anchor_body_name="waist_yaw_link",
        body_names=[
            "base_link",
            "left_leg_pelvic_roll_link",
            "left_leg_knee_pitch_link",
            "left_leg_ankle_roll_link",
            "right_leg_pelvic_roll_link",
            "right_leg_knee_pitch_link",
            "right_leg_ankle_roll_link",
            "waist_yaw_link",
            "left_shoulder_roll_link",
            "left_elbow_pitch_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_pitch_link",
            "right_wrist_yaw_link",
        ],
        body_names_all=[
            "base_link",
            "left_leg_pelvic_pitch_link",
            "right_leg_pelvic_pitch_link",
            "waist_yaw_link",
            "left_leg_pelvic_roll_link",
            "right_leg_pelvic_roll_link",
            "head_yaw_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_leg_pelvic_yaw_link",
            "right_leg_pelvic_yaw_link",
            "head_pitch_link",
            "left_shoulder_roll_link",
            "right_shoulder_roll_link",
            "left_leg_knee_pitch_link",
            "right_leg_knee_pitch_link",
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_leg_ankle_pitch_link",
            "right_leg_ankle_pitch_link",
            "left_elbow_pitch_link",
            "right_elbow_pitch_link",
            "left_leg_ankle_roll_link",
            "right_leg_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link"
        ],
    )
