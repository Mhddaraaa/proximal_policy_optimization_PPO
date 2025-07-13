from models.clipping_approach import clipping_Policy
from models.penalty_approach import kl_Policy
from models.reinforce import REINFORCE
import warnings
warnings.filterwarnings('ignore')
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--env-name",
                    default="CartPole-v1", help="gym environment full name")
parser.add_argument("-p", "--model-path",
                    default='', help="Pretrained model")
parser.add_argument("-e", "--num-episode",
                    type=int, default=500, help="How many episode it play the game")
parser.add_argument("-l", "--learning-rate",
                    type=float, default=0.001, help="Policy and value networks learning rate")
parser.add_argument("-nh", "--num-hidden",
                    type=int, default=64, help="Number of neural networks hidden layers")
parser.add_argument("-o", "--optimizer-step",
                    type=float, default=0.99, help="Adam optimizer schedule steps")
parser.add_argument("-c", "--continuous",
                    action='store_true', help="add this flag if the game is continuous")
parser.add_argument("-t", "--test-agent",
                    action='store_true', help="test agent and record it's palying")
parser.add_argument("-b", "--bootstrapping",
                    action='store_true', help="Calculate return reward using bootstrapping")
parser.add_argument("-u", "--update-steps",
                    type=int, default=10, help="How many times update the network each episode")

parser.add_argument("-m", "--update-mode",
                    default="clipping" , help="KL penaly or clipping objective")


args = parser.parse_args()

'''
I train the model in these three game:
"CartPole-v1": Discrete
"LunarLander-v3": Discrete
"LunarLanderContinuous-v3": continuous, if you want to trian this add -c flag
'''

env_name = args.env_name
model_path = args.model_path
n_episode = args.num_episode
lr = args.learning_rate
n_hidden = args.num_hidden
optStep = args.optimizer_step
# if these arguments are not specify in CLI they will be False
continuous = args.continuous
test_agent = args.test_agent
bootstrapping = args.bootstrapping
update_steps = args.update_steps
mode = args.update_mode # there are 2 modes: 1.kl, 2.clipping

if mode == 'kl':
    net = kl_Policy(env_name, continuous=False, n_hidden=n_hidden, lr=lr, optStep=optStep)
else:
    net = clipping_Policy(env_name, continuous=False, n_hidden=n_hidden, lr=lr, optStep=optStep)


gamma = 0.99
agent = REINFORCE(env_name,
                  net,
                  gamma,
                  n_episode,
                  continuous=False,
                  bootstrapping=bootstrapping,
                  model_path=model_path,
                  update_steps=update_steps,
                  mode=mode)

if test_agent and agent.load(model_path):
    print('>>> Testing the agent...')
    agent.test_agent(n_episode)
else:
    print('>>> Training the agent...')
    agent.train()
    agent.log()

