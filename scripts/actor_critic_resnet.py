#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from res_net import ResNet18Conv
import time

class ActorCriticResNet(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        encoder_output_dim=184,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        # ResNet feature extractor
        self.visual_encoder = ResNet18Conv(
            input_channel=1,  # Number of input channels for RGB or depth input
            pretrained=True,  # Use pretrained weights if needed
            mlp_input_dim=num_actor_obs,
            mlp_output_dim=128  # Output dimension of the MLP following ResNet
        )

        mlp_input_dim_a = encoder_output_dim
        mlp_input_dim_c = encoder_output_dim
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    def split_observations(self, observations):
        """
        Split the observations into two parts:
        - First 56 dimensions: directly passed to the actor/critic.
        - Last 128 dimensions: processed by the ResNet-MLP encoder.
        Args:
            observations (torch.Tensor): Input tensor with shape (B, 56+128).
        Returns:
            tuple(torch.Tensor, torch.Tensor): Normal observations and visual observations.
        """
        normal_obs = observations[:, :56]  # First 56 dimensions
        visual_obs = observations[:, 56:]  # Last 128 dimensions
        return normal_obs, visual_obs

    def extract_features(self, visual_obs):
        """
        Extract features using the ResNet18Conv encoder.
        Args:
            visual_obs (torch.Tensor): Input tensor with shape (B, 1, H, W).
        Returns:
            torch.Tensor: Encoded features with shape (B, 128).
        """
        visual_obs = visual_obs.reshape(-1, 1, 96, 96)
        return self.visual_encoder(visual_obs)
    
    def encode_observation(self, observations):
        normal_obs, visual_obs = self.split_observations(observations)
        visual_features = self.extract_features(visual_obs) 
        combined_features = torch.cat((normal_obs, visual_features), dim=-1)
        return combined_features

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """
        Update the action distribution based on observations.
        """
        mean = self.actor(self.encode_observation(observations))  # Compute action mean
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """
        Compute the mean action for inference (no sampling).
        """
        return self.actor(self.encode_observation(observations))
    
    def evaluate(self, critic_observations, **kwargs):
        """
        Evaluate the critic for the given observations.
        """
        return self.critic(self.encode_observation(critic_observations))
        
    def compute_encoder_loss(self, observations):
        """
        Compute auxiliary loss for the encoder.
        Args:
            observations (torch.Tensor): Visual observations passed through the encoder.
        Returns:
            torch.Tensor: Encoder loss.
        """
        _, visual_obs = self.split_observations(observations)  # Extract visual observations
        visual_features = self.extract_features(visual_obs)  # Pass through ResNet+MLP

        # L2 Regularization (Encourages small, stable feature values)
        encoder_loss = torch.mean(torch.norm(visual_features, p=2, dim=1))
        return encoder_loss


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
