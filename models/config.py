import random
import math
import copy
from tqdm.auto import tqdm
from collections import defaultdict, deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import adamw
from torch.distributions import Normal


from torch.autograd import Variable


import gymnasium
from IPython.core.display import HTML
from base64 import b64encode
from gym.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit
import os


def display_video(episode=0, video_width=600, video_dir= "/content/video"):
    video_path = os.path.join(video_dir, f"rl-video-episode-{episode}.mp4")
    video_file = open(video_path, "rb").read()
    decoded = b64encode(video_file).decode()
    video_url = f"data:video/mp4;base64,{decoded}"
    return HTML(f"""<video width="{video_width}"" controls><source src="{video_url}"></video>""")

def create_env(name, render_mode="rgb_array", record=False, video_folder='./video'):
    # render mode: "human", "rgb_array", "ansi")
    env = gymnasium.make(name, render_mode=render_mode)
    # env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: x % 50 == 0)
    if record:
        print('Recording video...!')
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env)
    return env