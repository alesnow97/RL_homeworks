from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu
from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: bool,
    ) -> Tuple[torch.Tensor, dict, dict]:
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        # TODO(student): modify the loss to implement CQL
        # Hint: `variables` includes qa_values and q_values from your CQL implementation

        q_values = variables["q_values"]
        first_term_loss = - self.cql_alpha * q_values.mean()

        # discrete action setting 
        qa_values = variables["qa_values"]
        second_term_loss = self.cql_temperature * self.cql_alpha * torch.logsumexp(qa_values / self.cql_temperature, axis=1).mean()

        action_probs = torch.exp(qa_values / self.cql_temperature) / torch.sum(torch.exp(qa_values / self.cql_temperature), axis=1, keepdim=True)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), axis=1)

        loss = loss + first_term_loss + second_term_loss + entropy.mean()

        return loss, metrics, variables
