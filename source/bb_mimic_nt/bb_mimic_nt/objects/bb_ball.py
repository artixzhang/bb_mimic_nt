from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from pathlib import Path

BB_BALL_USD = str(
    Path(__file__).resolve().parents[2]
    / "assets" / "objects" / "bb_ball" / "bb_ball.usd"
)

BB_BALL_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BB_BALL_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.25, 0.0, 0.85),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
    prim_path="/World/envs/env_.*/bb_ball",
)
