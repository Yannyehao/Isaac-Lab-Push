# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import torch.nn as nn

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

def adjust_model_input_layer(runner, original_input_dim, target_input_dim):
    model = runner.alg
    # Adjust the actor input layer
    actor_layer = model.actor_critic.actor[0]
    if isinstance(actor_layer, nn.Linear):
        with torch.no_grad():
            if actor_layer.in_features == target_input_dim:
                new_weight = actor_layer.weight[:, :original_input_dim].clone()
                actor_layer.weight = nn.Parameter(new_weight)
                actor_layer.in_features = original_input_dim
            elif actor_layer.in_features == original_input_dim:
                new_actor_layer = nn.Linear(original_input_dim, actor_layer.out_features)
                new_actor_layer.load_state_dict({'weight': actor_layer.weight, 'bias': actor_layer.bias})
                model.actor_critic.actor[0] = new_actor_layer

    # Adjust the critic input layer
    critic_layer = model.actor_critic.critic[0]
    if isinstance(critic_layer, nn.Linear):
        with torch.no_grad():
            if critic_layer.in_features == target_input_dim:
                new_weight = critic_layer.weight[:, :original_input_dim].clone()
                critic_layer.weight = nn.Parameter(new_weight)
                critic_layer.in_features = original_input_dim
            elif critic_layer.in_features == original_input_dim:
                new_critic_layer = nn.Linear(original_input_dim, critic_layer.out_features)
                new_critic_layer.load_state_dict({'weight': critic_layer.weight, 'bias': critic_layer.bias})
                model.actor_critic.critic[0] = new_critic_layer

def get_observation_shape_info(obs):
    """Automatically generate observation shape info based on the observation tensor."""
    observation_shape_info = []
    index = 0

    # Mapping the fields with their expected sizes
    field_names = [
        ("joint_pos", 9),
        ("joint_vel", 9),
        ("object_position", 3),
        ("target_object_position", 7),
        ("actions", 8)
    ]

    # Count the objects and targets by their respective field sizes
    num_objects = (obs.shape[1] - 18 - 8) // 10  # excluding joint_pos, joint_vel, and actions

    observation_shape_info.append(("joint_pos", (9,)))
    observation_shape_info.append(("joint_vel", (9,)))

    index = 18  # after joint_pos and joint_vel

    for i in range(num_objects):
        observation_shape_info.append((f"object{i+1}_position", (3,)))
        observation_shape_info.append((f"target{i+1}_object_position", (7,)))
        index += 10  # increment by the size of object_position (3) + target_object_position (7)

    observation_shape_info.append(("actions", (8,)))

    return observation_shape_info

def get_observation_slices(observation_shape_info):
    joint_pos_slice = slice(0, 9)
    joint_vel_slice = slice(9, 18)
    action_slice = slice(-8, None)
    
    object_slices = []
    target_slices = []
    
    index = 18  # starting index for object positions
    for i in range(1, len(observation_shape_info) // 2):  # assumes equal number of objects and targets
        object_slices.append(slice(index, index + 3))
        index += 3
        target_slices.append(slice(index, index + 7))
        index += 7
    
    return joint_pos_slice, joint_vel_slice, object_slices, target_slices, action_slice

def main():
    """Play with RSL-RL agent."""
    # Parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Create Isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    print(f"[DEBUG] Environment created with task: {args_cli.task}")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = "/home/bbboy/Documents/02Work/IsaacLab/logs/rsl_rl/franka_push/2024-08-07_20-22-57/model_1499.pt"

    # Load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    print("[DEBUG] Before adjustment:")
    print(ppo_runner.alg.actor_critic)
    
    # Get the initial observations to determine the shape
    obs, _ = env.get_observations()

    # Automatically determine observation shape info
    observation_shape_info = get_observation_shape_info(obs)

    # Adjust input layers before loading the model
    original_input_dim = 36
    target_input_dim = obs.shape[1]
    adjust_model_input_layer(ppo_runner, original_input_dim, target_input_dim)

    print("[DEBUG] After adjustment:")
    print(ppo_runner.alg.actor_critic)

    # Load model weights
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Re-adjust the input layers after loading the model
    adjust_model_input_layer(ppo_runner, original_input_dim, target_input_dim)

    print("[DEBUG] Model loaded and adjusted successfully:")
    print(ppo_runner.alg.actor_critic)
    # Obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.actor_critic, path=export_model_dir, filename="policy.onnx")

    print("[DEBUG] Policy exported successfully.")

    # Get observation slices based on the automatically retrieved shape info
    joint_pos_slice, joint_vel_slice, object_slices, target_slices, action_slice = get_observation_slices(observation_shape_info)
   
# Simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # Get the observations
            obs, _ = env.get_observations()
            
            block_moved_status = []
            distances = []

            # Check distances and moved status for all blocks
            for i in range(len(object_slices)):
                object_position = obs[:, object_slices[i]]
                target_position = obs[:, target_slices[i]]
                distance = torch.norm(object_position - target_position[:,:3] , dim=1)
                distances.append(distance)
                block_moved_status.append(torch.all(distance < 0.05))
                print(f"[DEBUG] Distance between block {i+1} and target: {distance}")
                print(f"[INFO] Block {i+1} moved status: {block_moved_status[i]}")

            # Update actions for blocks that haven't moved
            for i, moved in enumerate(block_moved_status):
                if not moved:
                    obs_for_model = torch.cat((
                        obs[:, joint_pos_slice],
                        obs[:, joint_vel_slice],
                        obs[:, object_slices[i]],
                        obs[:, target_slices[i]],
                        obs[:, action_slice]
                    ), dim=1)
                    obs_for_model = obs_for_model.to(agent_cfg.device)

                    actions = policy(obs_for_model)
                    print(f"[DEBUG] Actions for block {i+1} shape: {actions.shape}")

                    # Execute actions and get new observation data
                    obs, _, _, _ = env.step(actions)
                    break  # Only update one block at a time

            # If all blocks have moved, keep the environment active with zero actions
            if all(block_moved_status):
                obs, _, _, _ = env.step(torch.zeros_like(actions))
                print("[DEBUG] All blocks have reached their targets, keeping environment active.")

    # Close the simulator
    env.close()
    print("[INFO] Simulation ended and environment closed.")

       

if __name__ == "__main__":
    # Run the main function
    print("[INFO] Starting main function.")
    main()
    # Close sim app
    simulation_app.close()
    print("[INFO] Simulation app closed.")
