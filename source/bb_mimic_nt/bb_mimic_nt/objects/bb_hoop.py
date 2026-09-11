from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from pathlib import Path

BB_HOOP_USD = str(
    Path(__file__).resolve().parents[2]
    / "assets" / "objects" / "bb_hoop" / "bb_hoop.usd"
)

BB_HOOP_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/bb_hoop",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BB_HOOP_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(6.4, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
)
