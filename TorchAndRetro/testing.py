#!/usr/bin/env python3

import numpy as np
import retro
import torch
import tensorboard


from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
print("OK")

print("test")

env = retro.make('BalloonFight-Nes')
env.reset()
env.render()
