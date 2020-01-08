#!/usr/bin/env python3

import retro
'''
The idea behind this file is that I want to combine:
- balloon fight
- reinforcement (policy gradient)
- dense layers (hold off on convolutional neural net)
- Keras/Tensorflow (2.0?)


'''

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnLnLstmPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2, A2C

#env = DummyVecEnv([lambda: retro.make('BalloonFight-Nes', record='/gdrive/My Drive/science fair project/2019/BalloonFightRecord/')])
env = DummyVecEnv([lambda: retro.make('BalloonFight-Nes')])

modelname = 'balloonfightppo'
model = PPO2(CnnPolicy,env,n_steps=2048, verbose=1)
#model = PPO2.load("BalloonFightRecord/" + modelname+".pkl")
for saveloop in range(100):
  print("saveloop = "+str(saveloop))
  for learnloop in range(15):
    print("learnloop = "+str(learnloop))
    model.learn(total_timesteps=10000)
  print("saving "+str(modelname))
  model.save("BalloonFightRecord/" + modelname)

'''
obs = env.reset()
done = False
reward = 0

while not done:
  actions, _ = model.predict(obs)
  obs, rew, done, info = env.step(actions)
  reward += rew

  
print(rew)
'''