import copy
import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
from models.networks import PolicyNetwork,ValueNetwork
from models.config import create_env


# Dynamic KL Penalty (Beta Adjustment)
class kl_Policy():
    """
    PPO implementation.
    """
    def __init__(
            self, env_name, continuous=True, n_hidden=50, lr=0.05, optStep=0.9
    ):
        """
        Initialize the DQN agent.

        Args:
            env_name: OpenAI Gym environment.
            continuous: Whether the model is continous or discrete
            n_hidden (int): Number of hidden units in the neural network.
            lr (float): Learning rate for the optimizer.
            optStep (float): Step size factor for learning rate scheduler.
        """
        print('-- You are using KL penalty objecitve.')
        env = create_env(env_name)
        self.continuous = continuous
        self.action_dim = env.action_space.shape[0] if continuous else env.action_space.n  # Number of possible actions
        self.n_state = env.observation_space.shape[0] # Dimension of state space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize the primary and target networks
        self.policy_model = PolicyNetwork(self.n_state, self.action_dim, continuous, n_hidden).to(self.device)
        self.policy_model_old = copy.deepcopy(self.policy_model)
        self.value_model = ValueNetwork(self.n_state, n_hidden).to(self.device)

        # Set up the optimizer and learning rate scheduler
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr)
        self.policy_schedul = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=50, gamma=optStep)

        self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr)
        self.value_schedul = torch.optim.lr_scheduler.StepLR(self.value_optimizer, step_size=50, gamma=optStep)


    def predict(self, state):
        """
        Predict Q-values for the given state using the primary network.

        Args:
            state (Tensor): Input state.

        Returns:
            Tensor: Predicted Q-values.
        """
        if self.continuous:
            with torch.no_grad():
                mu, log_std = self.policy_model(state)  # Mean output from the network
                # log_std = torch.clamp(log_std, min=-20, max=2)
            sigma = torch.exp(log_std) + 1e-6  # Convert to standard deviation and Ensure sigma is not zero
            return Normal(mu, sigma)
        else:
            with torch.no_grad():
                logits, _ = self.policy_model(state)
                return nn.functional.softmax(logits, -1)


    def compute_advantage(self, states, returns):
        """
        Compute the advantage using the baseline (value network).

        Args:
            states (Tensor): Batch of states.
            next_states (Tensor): Batch of next_states.
            done (Tensor): Batch of done.
            returns (Tensor): Batch of discounted returns.

        Returns:
            Tensor: Advantage for each state-action pair.
        """
        with torch.no_grad():
            Value = self.value_model(states).squeeze()  # Predicted value

        return returns - Value


    def compute_policy_loss(self, states, actions, advantage, beta):

        if self.continuous:
            mu, log_std = self.policy_model(states)  # Mean output from the network
            # log_std = torch.clamp(log_std, min=-20, max=2)
            sigma = torch.exp(log_std) + 1e-6  # Convert to standard deviation and Ensure sigma is not zero
            dist = Normal(mu, sigma)
            log_probs_for_actions = dist.log_prob(actions).sum(dim=-1)  # Log-prob of the action

            with torch.no_grad():
                mu_old, log_std_old = self.policy_model_old(states)  # Mean output from the network
                sigma_old = torch.exp(log_std_old) + 1e-6
                dist_old = Normal(mu_old, sigma_old)
                log_probs_for_actions_old = dist_old.log_prob(actions).sum(dim=-1)

            kl_divergence = (
                log_std_old - log_std + (sigma**2 + (mu - mu_old)**2) / (2.0 * sigma_old**2) - 0.5
            ).sum(dim=-1).mean().item()

        else:
            logits, _ = self.policy_model(states)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = nn.functional.log_softmax(logits, -1)
            log_probs_for_actions = log_probs[range(len(actions)), actions]

            with torch.no_grad():
                logits_old, _ = self.policy_model_old(states)
                probs_old = nn.functional.softmax(logits_old, dim=-1)
                log_probs_old = nn.functional.log_softmax(logits_old, -1)
                log_probs_for_actions_old = log_probs_old[range(len(actions)), actions]

            kl_divergence = (probs_old * (log_probs_old - log_probs)).sum(dim=-1).mean().item()


        # Compute the ratio
        ratio = torch.exp(log_probs_for_actions - log_probs_for_actions_old)
        # Compute the penalty term
        penalty = 0.5 * ((ratio - 1) ** 2).mean()
        # Surrogate loss
        surrogate = (ratio * advantage).mean()
        # Final loss
        loss = -surrogate + beta * penalty

        return loss, kl_divergence


    def update(self, states, actions, returns, target_kl=0.01,
               initial_beta=0.01, update_steps=10):
        
        beta = initial_beta  # Start with an initial beta value
        policy_loss, value_loss = [], []
        for _ in range(update_steps):  # Update policy multiple times
            advantages = self.compute_advantage(states, returns)
            p_loss, kl_divergence = self.compute_policy_loss(states, actions, advantages, beta)
            self.policy_model_old.load_state_dict(self.policy_model.state_dict())

            # Backpropagate and update the policy model
            self.policy_optimizer.zero_grad()
            p_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=0.5)  # Gradient clipping
            self.policy_optimizer.step()
            policy_loss.append(p_loss.detach().cpu().item())

            # Update the value network separately
            for _ in range(update_steps // 2):  # Separate steps for value updates
                value = self.value_model(states).squeeze()
                v_loss = nn.functional.mse_loss(value, returns)
                self.value_optimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=0.5)  # Gradient clipping
                self.value_optimizer.step()
                value_loss.append(v_loss.detach().cpu().item())

            # Dynamically adjust beta based on KL divergence
            if kl_divergence > 1.5 * target_kl:
                beta = min(2 * beta, 10.0)  # Increase beta, but cap it at 10.0
            elif kl_divergence < 0.5 * target_kl:
                beta = max(0.5 * beta, 1e-5)  # Decrease beta, but keep it above 1e-5
        
        return np.mean(policy_loss), np.mean(value_loss)