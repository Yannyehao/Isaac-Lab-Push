# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


# object1 position in robot root frame
def object_position_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

# object2 position in robot root frame
def object2_position_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("object2")
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object2: RigidObject = env.scene[object2_cfg.name]
    object_pos_w2 = object2.data.root_pos_w[:, :3]
    object_pos_b2, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w2
    )
    return object_pos_b2


# object3 position in robot root frame
def object3_position_in_robot_root_frame(
    env: RLTaskEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object3_cfg: SceneEntityCfg = SceneEntityCfg("object3")
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object3: RigidObject = env.scene[object3_cfg.name]
    object_pos_w3 = object3.data.root_pos_w[:, :3]
    object_pos_b3, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w3
    )
    return object_pos_b3
