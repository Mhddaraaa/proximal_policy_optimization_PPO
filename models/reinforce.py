from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.config import create_env
from IPython.core.display import clear_output
import os


class REINFORCE():
    """
    A policy gradient agent implementing the REINFORCE algorithm.

    The agent learns to maximize the expected return using policy gradient updates.
    """

    def __init__(
            self, env_name, network, gamma, n_episode, continuous,
            bootstrapping=False, model_path=None, update_steps=10, mode='clipping'
    ):
        """
        Initialize the REINFORCE agent.

        Args:
            en_name: The name of the environment (OpenAI Gym compatible).
            network: The neural network used for policy and value approximation and updates.
            gamma: Discount factor for future rewards.
            n_episode: Total number of episodes for training.
            continuous: Whether the model is continous or discrete
            bootstrapping: Use bootstrapping for next state value
            model_path: Policy and value pretrian models path
            update_steps: Update network update_steps times in each update
            mode: Policy update in PPO has two approach: 1.Clipping, 2. KL penalty
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

        # Environment and agent parameters
        self.env_name = env_name
        self.mode = mode
        self.env = create_env(env_name)
        self.network = network  # Policy network
        self.gamma = gamma  # Discount factor for rewards
        self.n_episode = n_episode  # Total number of training episodes
        self.epi = 0  # Current episode index
        self.trajectory = []  # Stores state-action-reward trajectories
        self.bootstrapping = bootstrapping
        self.continuous = continuous

        # Metrics to track training progress
        self.length_episode = [] # Track number of steps per episode
        self.total_reward_episode = []  # Track total rewards per episode
        self.policy_loss = []
        self.value_loss = []
        self.update_steps = update_steps
        self.best_reward = float('-inf')

        if model_path:
            self.load(model_path)

    def load(self, path):

        policy_path = os.path.join(path, f'{self.env_name}_{self.mode}_policy_network.pth')
        value_path = os.path.join(path, f'{self.env_name}_{self.mode}_value_network.pth')

        if os.path.isfile(policy_path) and os.path.isfile(value_path):
            print('Loading policy and value models...')

            checkpoint = torch.load(policy_path, map_location=self.device)
            self.network.policy_model.load_state_dict(checkpoint['model_state_dict'])
            self.network.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.network.policy_schedul.load_state_dict(checkpoint['lr_sched_state_dict'])

            checkpoint = torch.load(value_path, map_location=self.device)
            self.network.value_model.load_state_dict(checkpoint['model_state_dict'])
            self.network.value_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.network.value_schedul.load_state_dict(checkpoint['lr_sched_state_dict'])
            return True
        else:
            print('policy and value models files do not exist...')
            return False

    def test_agent(self, num_test_episodes):
        # Evaluate the agent's performance over a specified number of test episodes
        ep_rets, ep_lens = [], []  # Episode returns and lengths
        env = create_env(self.env_name, record=True)
        for j in range(num_test_episodes):
            state, _ = env.reset()  # Reset the environment
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            done, truncate, rewards, length = False,False, 0, 0
            while True:
                if self.continuous:
                    dist = self.network.predict(state)
                    action = dist.sample().cpu().numpy()
                else:
                    # Predict action probabilities using the policy network
                    probs = self.network.predict(state)
                    # Sample an action based on the probabilities
                    action = torch.multinomial(probs, 1).item()

                state, reward, done, truncate, _ = env.step(action)
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                rewards += reward
                length += 1
                if done or truncate:
                    break
            ep_rets.append(rewards)  # Record the total reward
            ep_lens.append(length)  # Record the episode length

        print(f"Average reward: {np.mean(ep_rets):.2f} - "
              f"Average length of episodes: {np.mean(ep_lens)}")  # print average reward and length


    def train(self):
        """
        Train the agent using the REINFORCE algorithm.
        """
        print(f"Available device: {self.device} - "
              f"Environment name: {self.env_name} - "
              f"Highest possible score: {self.env.spec.reward_threshold}")
        pbar = tqdm(range(self.n_episode))  # Progress bar for training episodes

        for episode in pbar:
            # Play one episode and collect trajectory data
            self.play()

            # Extract states, actions, and rewards from the trajectory
            states, actions, rewards, next_states, done = zip(*self.trajectory)

            # Convert states and actions to tensors
            states = torch.stack(states).to(self.device)  # Batch of states
            actions = torch.tensor(actions).to(self.device)  # Actions taken
            next_states = torch.stack(next_states).to(self.device)  # Batch of next_states
            done = torch.stack(done).to(self.device)  # Batch of done

            # Compute discounted returns and normalize them
            returns = self.compute_discounted_returns(rewards, next_states, done)

            # Update the policy network using the collected trajectory
            p_loss, v_loss = self.network.update(states, actions, returns, update_steps=self.update_steps)
            self.policy_loss.append(p_loss)
            self.value_loss.append(v_loss)

            # Update the progress bar with learning rate and recent average rewards
            avg_reward = np.mean(self.total_reward_episode[max(0, episode - 50):episode + 1])
            pbar.set_description(f"α Neural network: {round(self.network.policy_optimizer.param_groups[0]['lr'], 6)} |"
                                 f" Avg Reward: {avg_reward:.2f}")

            if self.total_reward_episode[self.epi - 1] > self.best_reward:
                self.best_reward = self.total_reward_episode[self.epi - 1]

                torch.save(
                    {
                        'model_state_dict': self.network.policy_model.state_dict(),
                        'optimizer_state_dict': self.network.policy_optimizer.state_dict(),
                        'lr_sched_state_dict': self.network.policy_schedul.state_dict()
                    }, f'{self.env_name}_{self.mode}_policy_network.pth'
                )

                torch.save(
                    {
                        'model_state_dict': self.network.value_model.state_dict(),
                        'optimizer_state_dict': self.network.value_optimizer.state_dict(),
                        'lr_sched_state_dict': self.network.value_schedul.state_dict()
                    }, f'{self.env_name}_{self.mode}_value_network.pth'
                )


    def compute_discounted_returns(self, rewards, next_states, done):
        """
        Compute discounted returns for a single trajectory.

        Args:
            rewards (list): Rewards obtained during the trajectory.

        Returns:
            list: Discounted returns for each time step.
        """
        returns = []
        G = 0  # Initialize the discounted return
        if self.bootstrapping:
            # High bias low variance
            with torch.no_grad():
                V_next = self.network.value_model(next_states).squeeze()
            V_next = (V_next * (1 - done)).cpu().numpy()
            rewards = rewards + self.gamma * V_next

        # High variance low bias if not bootstrapping
        for r in reversed(rewards):
            G = r + self.gamma * G  # Compute the discounted return
            returns.insert(0, G)  # Insert at the beginning to reverse the order

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize returns for stability

        return returns

    def play(self):
        """
        Play one episode of the environment.
        """
        # Reset the environment and get the initial state
        state, _ = self.env.reset()
        state = torch.tensor(state, device=self.device)  # Convert state to tensor
        done = False  # Indicates if the episode is over
        truncate = False
        length_episode = 0  # Track the episode length
        total_reward_episode = 0  # Track total rewards for the episode
        self.trajectory = []  # Clear the trajectory for the new episode

        while True:

            if self.continuous:
                dist = self.network.predict(state)
                action = dist.sample().cpu().numpy()
            else:
                # Predict action probabilities using the policy network
                probs = self.network.predict(state)
                # Sample an action based on the probabilities
                action = torch.multinomial(probs, 1).item()

            # Take the action in the environment and observe the outcome
            next_state, reward, done, truncate, _ = self.env.step(action)


            # Update episode statistics
            length_episode += 1
            total_reward_episode += reward

            # Convert the next state to a tensor
            next_state = torch.tensor(next_state, device=self.device)
            done = torch.tensor(done, dtype=torch.long)

            # Store the experience (state, action, reward) in the trajectory
            self.trajectory.append((state, action, reward, next_state, done))

            if done or truncate:
                break

            # Update the current state
            state = next_state


        # Increment the episode counter and adjust the learning rate schedule
        self.epi += 1
        self.length_episode.append(length_episode)
        self.total_reward_episode.append(total_reward_episode)
        self.network.policy_schedul.step()
        self.network.value_schedul.step()
    
    def log(self):
        """
        Log training statistics and plot metrics.
        """
        fig = plt.figure(figsize=(16, 4))

        # Plot total reward per episode
        fig.add_subplot(1, 4, 1)
        plt.plot(self.total_reward_episode)
        plt.title('Episode reward over time')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')

        # Plot episode length over time
        fig.add_subplot(1, 4, 2)
        plt.plot(self.length_episode)
        plt.title('Episode length over time')
        plt.xlabel('Episode')
        plt.ylabel('Length')

        # Plot policy loss over episodes
        fig.add_subplot(1, 4, 3)
        plt.plot(self.policy_loss)
        plt.title('Policy loss over time')
        plt.xlabel('Episode')
        plt.ylabel('Policy loss')

        # Plot value loss over episodes
        fig.add_subplot(1, 4, 4)
        plt.plot(self.value_loss)
        plt.title('Value loss over time')
        plt.xlabel('Episode')
        plt.ylabel('Value loss')

        plt.show()
