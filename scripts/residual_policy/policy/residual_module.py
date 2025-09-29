# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from .encoder import TransformerEncoder

class ResidualModule(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_encoder_obs,
        num_time_steps,
        num_encoder_output,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        encoder_d_model=128,
        encoder_nhead=8,
        encoder_num_layers=2,
        burnin_epochs=0,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ResidualModule.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)


        self.num_encoder_obs = num_encoder_obs
        self.num_time_steps = num_time_steps
        self.encoder_total_dim = num_time_steps * num_encoder_obs

        mlp_input_dim_a = int((num_actor_obs - self.encoder_total_dim) + num_encoder_output)
        mlp_input_dim_c = int((num_critic_obs - self.encoder_total_dim) + num_encoder_output)
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
        # zero init the last layer of the actor network
        with torch.no_grad():
            self.actor[-1].weight.zero_()
            self.actor[-1].bias.zero_()

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

        # Encoder
        self.encoder = TransformerEncoder(num_encoder_obs, encoder_d_model, encoder_nhead, encoder_num_layers, num_encoder_output, num_time_steps)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Encoder: {self.encoder}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        self.burnin_epochs = burnin_epochs
        self.current_epoch = 0
        self.in_burnin = burnin_epochs > 0
        if self.in_burnin:
            print(f"[INFO] Freeze Actor Network for Burn-in Epoch {self.burnin_epochs}")
            self._freeze_actor()

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
    
    def _freeze_actor(self):
        for param in self.actor.parameters():
            param.requires_grad = False
        if hasattr(self, 'std'):
            self.std.requires_grad = False
            self._original_std = self.std.data.clone()
            self.std.data.fill_(1e-6)
        elif hasattr(self, 'log_std'):
            self.log_std.requires_grad = False
            self._original_log_std = self.log_std.data.clone()
            self.log_std.data.fill_(torch.log(torch.tensor(1e-6)))

    def _unfreeze_actor(self):
        for param in self.actor.parameters():
            param.requires_grad = True
        if hasattr(self, 'std'):
            self.std.requires_grad = True
            self.std.data.copy_(self._original_std)
        elif hasattr(self, 'log_std'):
            self.log_std.requires_grad = True
            self.log_std.data.copy_(self._original_log_std)

    
    def _extract_encoder_obs(self, observations):
        # extract encoder observations
        batch_size = observations.shape[0]
        encoder_obs = observations[:, -self.encoder_total_dim:]  # (B, 105)
        encoder_obs = encoder_obs.view(batch_size, self.num_time_steps, self.num_encoder_obs)
        return encoder_obs
    
    def _extract_regular_obs(self, observations):
        # extract regular observations
        return observations[:, :-self.encoder_total_dim]
    
    def _prepare_obs_input(self, observations):
        # extract and encode encoder observations
        encoder_obs = self._extract_encoder_obs(observations)  # (B, 5, 21)
        encoded_obs = self.encoder(encoder_obs)  # (B, 32)
        
        # extract regular observations
        regular_obs = self._extract_regular_obs(observations)  # (B, regular_dim)
        
        # concatenate
        return torch.cat([regular_obs, encoded_obs], dim=-1)
    

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
        # prepare observations
        actor_input = self._prepare_obs_input(observations)
        # compute mean
        mean = self.actor(actor_input)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_input = self._prepare_obs_input(observations)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_input = self._prepare_obs_input(critic_observations)
        value = self.critic(critic_input)
        return value
    
    def step_burnin(self):
        self.current_epoch += 1
        if self.in_burnin and self.current_epoch >= self.burnin_epochs:
            self._unfreeze_actor()
            self.in_burnin = False
            print(f"Burn-in ended at epoch {self.current_epoch}")


    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True

