# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
 
import argparse
 
from isaaclab.app import AppLauncher
 
# 使用parser在运行脚本时候可以直接传入超参数
parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
# SimulationApp 超参数列表https://docs.omniverse.nvidia.com/py/isaacsim/source/isaacsim.simulation_app/docs/index.html?highlight=simulationapp#isaacsim.simulation_app.SimulationApp
parser.add_argument(
    "--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280"
)
parser.add_argument(
    "--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720"
)
 
# 添加parser到launcher里面
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
 
###################################
#截止到这里是如何通过IsaacLab加载IsaacSim simulator并设置特定超参数的常规流程。
#因为Sim是仿真#器，Lab只是训练框架，所以加载Sim是必不可少的。
#大家可以熟记整个流程，有助于后续的开发，特别是headless模式，设置保存的ideo的时候。
###################################
 
 
###################################
#下面的代码是设计环境的流程，非常建议大家反复观看理解，对后续看代码非常有帮助。
#如之前所说，机器人+learning综合了仿真和训练两个流程，再加上代码追求解耦模块化，
#在看大项目时候很容易淹没在各种各样的脚本文件抓不到主干思路。
#虽然后续的代码会将design_scene模块化，但是整体思路是相似的。
###################################
 
import isaacsim.core.utils.prims as prim_utils
 
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
 
 
def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
 
    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))
    
 
    ###################################    
    #到这里上面都是加载地面，灯光啊之类的东西，不是特别重要，复制粘贴就行了
    #要注意IsaacLab思路很OOP，把各个object比如地面，灯光，刚体，变形体等封装成类
    #设置超参数，调用，然后用.func加载，一般是这个思路。
    ###################################
 
    # create a new xform prim for all objects to be spawned under
    # prim是IsaacSim的特色，类似于一个筐，筐里面可以装rigidbody，articulation各种类
    # 需要的时候就直接拎筐，打开筐，从筐里取出自己需要的，所以创建prim基本上是必备的。
    # 而且看代码也可以发现这个是直接从isaacsim调用的，不是从isaaclab包
    prim_utils.create_prim("/World/Objects", "Xform")
 
    # 先生成物体的设置，再用.func()加载。基本的形状都有封装，建议大家多F12看看源代码
    cfg_cone = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0, 1.0, 1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0, -1.0, 1.0))
 
    # spawn a green cone with colliders and rigid body
    # 这里是加载刚体的属性，如果没有这些属性的话默认是None，不具备属性，无碰撞也不受重力影响，
    # 建议大家运行一下对比效果，更加直观
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    cfg_cone_rigid.func(
        "/World/Objects/ConeRigid", 
        cfg_cone_rigid, 
        translation=(-0.2, 0.0, 2.0), 
        orientation=(0.5, 0.0, 0.5, 0.0)
    )
 
    # spawn a blue cuboid with deformable body
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
        size=(0.2, 0.5, 0.2),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func(
        "/World/Objects/CuboidDeformable", 
        cfg_cuboid_deformable, 
        translation=(0.15, 0.0, 2.0)
    )
 
    # spawn a usd file of a table into the scene
    # 这个UsdFileCfg很重要，建议大家反复看源代码，熟练掌握，因为自定义环境都需要用自己的模型
    # 很少用自带的cone，cube等简单形状，这些单纯就是入门用的。
    # 熟悉掌握这个类对物理仿真很有帮助，可以避免很多莫名其妙的报错。
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    
    cfg.func(
        "/World/Objects/Table", 
        cfg, 
        translation=(0.0, 0.0, 1.05)
    )
 
 
 
###################################    
#到这里基本上就算设计环境的流程了，大家掌握思路，后续看一些lab_task会轻松很多
#下面的就是如何打开仿真器，加载设计的环境了。
###################################
 
 
def main():
    """Main function."""
 
    # Initialize the simulation context
    # 这个sim_cfg也很重要，重力，物理推理时间间隔，渲染时间等等，对训练都有影响，建议多看源码
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
 
    # Design scene by adding assets to it
    design_scene()
 
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
 
    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()
 
 
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()