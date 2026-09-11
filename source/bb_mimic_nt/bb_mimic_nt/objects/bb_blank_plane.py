import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg

BB_BLANK_PLANE_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,

    physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=0.8,
        restitution=0.0,
    ),

    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.8, 0.5, 0.2),
        roughness=0.4,
        metallic=0.0,
    ),
    debug_vis=False,
)