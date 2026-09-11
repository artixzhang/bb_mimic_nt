from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

BB_HOOP_FLOATING_USD = str(
    Path(__file__).resolve().parents[2]
    / "assets" / "objects" / "bb_hoop" / "bb_hoop_floating.usd"
)

BB_HOOP_CENTER_OFFSET = (2.17577, 0.0, 3.02)
BB_HOOP_ROOT_ROTATION = (0.0, 0.0, 0.0, 1.0)

BB_HOOP_FLOATING_CFG = AssetBaseCfg(
    prim_path="/World/envs/env_.*/bb_hoop_floating",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BB_HOOP_FLOATING_USD,
        activate_contact_sensors=False,
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # 180 degrees around +Z. Isaac Lab quaternion order is WXYZ.
        rot=BB_HOOP_ROOT_ROTATION,
    ),
)
