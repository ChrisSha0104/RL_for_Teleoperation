# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.assets import AssetBaseCfg, AssetBase
from omni.isaac.lab.assets import RigidObjectCfg, RigidObject
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

import pdb




@configclass
class FrankaCubeResidualFixedEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6.6666 #3.31 ~ 200 timesteps  # 8.3333 = 500 timesteps
    decimation = 2
    action_space = 8 # waypoint
    observation_space = 3*8+4*8 # last 3 wp + last 3 expert wp
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            # pos=(1.0, 0.0, 0.0),
            # rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80, # changed stiffness
                damping=80,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80,
                damping=80,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        # soft_joint_pos_limit_factor=1.0,
    )

    # cube
    # cube = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Cube", # name in simulator
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd", 
    #         scale=(0.8, 0.8, 0.8)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.5, 0.0, -1.05),
    #         rot=(0,0,0,1.0),
    #         )
    # )
    cube = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, 0), 
                rot=(0, 0, 0, 1),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd", 
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 7.5
    dof_velocity_scale = 0.1
    minimal_height = 0.10
    std = 0.1

    # reward scales #TODO: fix rewards
    # dist_reward_scale = 0.5
    # rot_reward_scale = 1.5
    # action_penalty_scale = 0.05
    # finger_reward_scale = 2.0

    # cube residual rewards v2
    # tracking_reward_scale = -2.0
    # height_reward_scale = 20.0 -> improportional!
    # ee_dist_reward_scale = 2.0
    # residual_penalty_scale = -0.1

    # # cube residual rewards v3
    # tracking_reward_scale = -1.5
    # height_reward_scale = 5.0
    # ee_dist_reward_scale = 2.0
    # residual_penalty_scale = -0.5

    # cube residual rewards v3
    tracking_reward_scale = -1.0 #-1.0
    height_reward_scale = 2.0
    ee_dist_reward_scale = 0.5
    residual_penalty_scale = -0.5 # equivalent to tracking rew?


class FrankaCubeResidualFixedEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: FrankaCubeResidualFixedEnvCfg

    def __init__(self, cfg: FrankaCubeResidualFixedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.demo_path = "/home/shuosha/Desktop/IsaacLab/IsaacLab/source/standalone/workflows/rsl_rl/wp_traj1/wp_traj_cube_bar2.txt"

        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self._robot.find_joints("panda_finger_joint2")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_link7")),
            self.device,
        )
        lfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_leftfinger")),
            self.device,
        )
        rfinger_pose = get_env_local_pose(
            self.scene.env_origins[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_rightfinger")),
            self.device,
        )

        finger_pose = torch.zeros(7, device=self.device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        robot_local_grasp_pose_rot, robot_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )
        robot_local_pose_pos += torch.tensor([0, 0.04, 0], device=self.device)
        self.robot_local_grasp_pos = robot_local_pose_pos.repeat((self.num_envs, 1))
        self.robot_local_grasp_rot = robot_local_grasp_pose_rot.repeat((self.num_envs, 1))

        cube_local_grasp_pose = torch.tensor([0.3, 0.01, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device)
        self.cube_local_grasp_pos = cube_local_grasp_pose[0:3].repeat((self.num_envs, 1))
        self.cube_local_grasp_rot = cube_local_grasp_pose[3:7].repeat((self.num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.cube_inward_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )
        self.cube_up_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat(
            (self.num_envs, 1)
        )

        self.hand_link_idx = self._robot.find_bodies("panda_link7")[0][0]
        self.left_finger_link_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_link_idx = self._robot.find_bodies("panda_rightfinger")[0][0]
        # self.drawer_link_idx = self._cabinet.find_bodies("drawer_top")[0][0]

        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.cube_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.cube_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # init buffer for robot last 3 actions
        self.robot_wp = torch.zeros((self.num_envs, 8), device=self.device).repeat(1,3)

        self.time_step_per_env = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        with open(self.demo_path, "r") as file:
            lines = file.readlines()
        
        demo_traj_list = []
        for line in lines:
            data = eval(line.strip())
            demo_traj_list.append(data)

        # print("demo traj: ", demo_traj_list[:5])
        # self.demo_traj -> (num_envs, traj_length*8)
        '''
        \[
        ee_1, ee_2, ..., ee_n
        .
        .
        .
        ee_1, ee_2, ..., ee_n
        \]
        '''
        demo_traj_flatten = [item for sublist in demo_traj_list for item in sublist]

        self.demo_traj = torch.tensor(demo_traj_flatten, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        # print("demo traj shape: ", self.demo_traj.shape)

        self.offsets = torch.arange(8, device=self.device)
        self.row_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand(-1, 8)

        # init buffer for current & last 3 demo actions
        self.demo_wp = torch.zeros((self.num_envs, 8), device=self.device).repeat(1,4)

        self.joint_ids = list(range(self._robot.num_joints))
        # print("joint ids: ", self.joint_ids)

        self.body_ids = list(range(self._robot.num_bodies))
        # print("body names: ", self._robot.body_names)

        # pdb.set_trace()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._cube = RigidObject(self.cfg.cube)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["cube"] = self._cube

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, ee_residual: torch.Tensor):
        self.curr_col_idx = self.time_step_per_env.unsqueeze(1)*8 + self.offsets

        # for RP, the arg actions is the residual!
        self.ee_residual = ee_residual.clone() # .clamp(-1.0, 1.0) # TODO: check if this is necessary
        self.joint_pos = self.get_joint_pos_from_ee_pos(self.demo_traj[self.row_idx, self.curr_col_idx]+ self.ee_residual)
        self.robot_dof_targets[:] = torch.clamp(self.joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        self.time_step_per_env += 1

        # print("actions", actions)

        # self.actions = actions.clone().clamp(-1.0, 1.0) 
        # targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        # self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def step(self, ee_residual):
        # print("ee_residual_output: ", ee_residual)
        _return = super().step(ee_residual)

        # update robot wp
        self.robot_wp = torch.roll(self.robot_wp, shifts = 8, dims = 1)

        # ee and root pose in world frame
        curr_ee_pose_w = self._robot.data.body_state_w[:,8,0:7]
        curr_root_pose_w = self._robot.data.root_state_w[:, 0:7]

        # ee pose in base (local) frame
        curr_ee_pos_b, curr_ee_quat_b = subtract_frame_transforms(
                curr_root_pose_w[:, 0:3], curr_root_pose_w[:, 3:7], curr_ee_pose_w[:, 0:3], curr_ee_pose_w[:, 3:7]
            )

        curr_finger_status = ((self.robot_dof_targets[:,-1] < 0.035) * (self.robot_dof_targets[:,-2] < 0.035)).unsqueeze(1)
        curr_ee_pos_combined_b = torch.cat((curr_ee_pos_b, curr_ee_quat_b, curr_finger_status), dim=-1)
        self.robot_wp[:, :8] = curr_ee_pos_combined_b

        # update demo wp
        self.curr_col_idx = self.time_step_per_env.unsqueeze(1)*8 + self.offsets
        self.demo_wp = torch.roll(self.demo_wp, shifts = 8, dims = 1)
        self.demo_wp[:, :8] = self.demo_traj[self.row_idx, self.curr_col_idx]

        # print("average residual: ", torch.sum(torch.norm(self.robot_wp[:,:8]-self.demo_wp[:,8:16], p=2, dim=-1))/self.num_envs)

        return _return


    # post-physics step calls 
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._cube.data.body_pos_w[:,0,2] >= 0.5 #* (torch.abs(self._cube.data.body_pos_w[:,0,0]-self.scene.env_origins[:, 0]-0.5) > 0.1) * (torch.abs(self._cube.data.body_pos_w[:,0,1]-self.scene.env_origins[:, 1]) > 0.1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()
        self.robot_left_finger_pos = self._robot.data.body_pos_w[:, self.left_finger_link_idx]
        self.robot_right_finger_pos = self._robot.data.body_pos_w[:, self.right_finger_link_idx]

        return self._compute_rewards()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # robot state -> static reset
        joint_pos = self._robot.data.default_joint_pos[env_ids] #+ sample_uniform(
        #     -0.125,
        #     0.125,
        #     (len(env_ids), self._robot.num_joints),
        #     self.device,
        # )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # cube state
        reseted_root_states = self._cube.data.default_root_state.clone()
        reseted_root_states[env_ids,:3] += self.scene.env_origins[env_ids,:]
        self._cube.write_root_state_to_sim(root_state=reseted_root_states, env_ids=env_ids)
        self._cube.reset(env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)

        self.time_step_per_env[env_ids] = 0
        self.robot_wp[env_ids, :] = 0
        self.demo_wp[env_ids, :] = 0
        self.demo_wp[env_ids,:8] = self.demo_traj[env_ids,:8]

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.robot_wp,
                self.demo_wp,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)} #TODO: check range

    # auxiliary methods
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        hand_pos = self._robot.data.body_pos_w[env_ids, self.hand_link_idx]
        hand_rot = self._robot.data.body_quat_w[env_ids, self.hand_link_idx]
        cube_pos = self._cube.data.body_pos_w[env_ids,0,:]
        cube_rot = self._cube.data.body_quat_w[env_ids,0,:]
        (
            self.robot_grasp_rot[env_ids],
            self.robot_grasp_pos[env_ids],
            self.cube_grasp_rot[env_ids],
            self.cube_grasp_pos[env_ids],
        ) = self._compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.robot_local_grasp_rot[env_ids],
            self.robot_local_grasp_pos[env_ids],
            cube_rot,
            cube_pos,
            self.cube_local_grasp_rot[env_ids],
            self.cube_local_grasp_pos[env_ids],
        )

    def _compute_rewards(self):
        # # distance from hand to the drawer
        # d = torch.norm(self.robot_grasp_pos - self.cube_grasp_pos, p=2, dim=-1)
        # dist_reward = 1.0 / (1.0 + d**2)
        # dist_reward *= dist_reward
        # dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

        # # # regularization on the actions (summed for each environment)
        # # action_penalty = torch.sum(actions**2, dim=-1)

        # # penalty for distance of each finger from the cube
        # lfinger_dist = self.robot_left_finger_pos[:, 2] - self.cube_grasp_pos[:, 2]
        # rfinger_dist = self.cube_grasp_pos[:, 2] - self.robot_right_finger_pos[:, 2]
        # finger_dist_penalty = torch.zeros_like(lfinger_dist)
        # finger_dist_penalty += torch.where(lfinger_dist < 0, lfinger_dist, torch.zeros_like(lfinger_dist))
        # finger_dist_penalty += torch.where(rfinger_dist < 0, rfinger_dist, torch.zeros_like(rfinger_dist))

        # difference between robot actions and demo actions
        tracking_dif = torch.norm(self.robot_wp[:,:8] - self.demo_wp[:,8:16], p=2, dim=-1)
        tracking_reward = tracking_dif

        # height of the cube
        height_reward = torch.where(self._cube.data.root_pos_w[:, 2] > self.cfg.minimal_height, 1.0, 0.0) * (1 + self._cube.data.root_pos_w[:, 2])

        # regularization on residuals
        residual_penalty = torch.sum(self.ee_residual**2, dim=-1)

        # reward for reaching cube

        # Target object position: (num_envs, 3)
        cube_pos_w = self._cube.data.root_pos_w
        # End-effector position: (num_envs, 3)
        ee_w = self._robot.data.body_state_w[:,8,0:3]
        # Distance of the end-effector to the object: (num_envs,)
        object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
        ee_dist_reward = 1 - torch.tanh(object_ee_distance / self.cfg.std)

        rewards = (
            # self.cfg.dist_reward_scale * dist_reward
            # + self.cfg.finger_reward_scale * finger_dist_penalty
            # - action_penalty_scale * action_penalty
            + self.cfg.tracking_reward_scale * tracking_reward
            + self.cfg.height_reward_scale * height_reward
            + self.cfg.residual_penalty_scale * residual_penalty
            + self.cfg.ee_dist_reward_scale * ee_dist_reward
        )

        rewards = torch.where(self._cube.data.root_pos_w[:, 2] > 0.15, rewards + 0.25, rewards)
        rewards = torch.where(self._cube.data.root_pos_w[:, 2] > 0.20, rewards + 0.25, rewards)

        self.extras["log"] = {
            # "dist_reward": (self.cfg.dist_reward_scale * dist_reward).mean(),
            # # "action_penalty": (-action_penalty_scale * action_penalty).mean(),
            # "left_finger_distance_reward": (self.cfg.finger_reward_scale * lfinger_dist).mean(),
            # "right_finger_distance_reward": (self.cfg.finger_reward_scale * rfinger_dist).mean(),
            # "finger_dist_penalty": (self.cfg.finger_reward_scale * finger_dist_penalty).mean(),
            "tracking_reward": (self.cfg.tracking_reward_scale * tracking_reward).mean(),
            "height_reward": (self.cfg.height_reward_scale * height_reward).mean(),
            "residual_penalty": (self.cfg.residual_penalty_scale * residual_penalty).mean(),
            "ee_dist_reward": (self.cfg.ee_dist_reward_scale * ee_dist_reward).mean(),
            "tracking_dif": (tracking_dif).mean(),
        }

        return rewards

    def _compute_grasp_transforms(
        self,
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
        cube_rot,
        cube_pos,
        cube_local_grasp_rot,
        cube_local_grasp_pos,
    ):
        # import pdb
        # pdb.set_trace()
        global_franka_rot, global_franka_pos = tf_combine(
            hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos
        )
        global_cube_rot, global_cube_pos = tf_combine(
            cube_rot, cube_pos, cube_local_grasp_rot, cube_local_grasp_pos
        )

        return global_franka_rot, global_franka_pos, global_cube_rot, global_cube_pos
    
    def get_joint_pos_from_ee_pos(self, ee_goal):
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        ik_commands = ee_goal[:, :-1] #(num_envs, 8)
        diff_ik_controller.reset()
        diff_ik_controller.set_command(ik_commands)
        
        if self._robot.is_fixed_base:
            ee_jacobi_idx = 6
        else:
            ee_jacobi_idx = 5
        
        jacobian = self._robot.root_physx_view.get_jacobians()[:,ee_jacobi_idx,:, self.joint_ids[:7]] #(num_envs, 6, 7)
        ee_pose_w = self._robot.data.body_state_w[:, 8, 0:7] # (num_envs, 11, 7) -> (num_envs, 7) #TODO: hardcoded to panda hand
        root_pose_w = self._robot.data.root_state_w[:, 0:7] # (num_envs, 7)
        joint_pos = self._robot.data.joint_pos[:,self.joint_ids[:7]] # (num_envs, 7)

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        joint_pos_des_arm = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        finger_joint_pos_des = torch.zeros((self.num_envs, 2), device=self.device) + 0.01
        open_idx = (ee_goal[:, -1] < 1)
        finger_joint_pos_des[open_idx] = 0.04

        joint_pos_des = torch.cat((joint_pos_des_arm, finger_joint_pos_des), dim=-1)

        return joint_pos_des
