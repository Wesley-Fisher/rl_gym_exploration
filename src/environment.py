import numpy as np
from typing import Tuple, List

import tensorflow as tf

import gym

# Can check some outside info: https://bytepawn.com/solving-openai-gym-classic-control-problems-with-pytorch.html
envs = ['CartPole-v0', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v1', 'MountainCarContinuous-v0']
min_episodes = {'CartPole-v0': 100, 'MountainCar-v0': 100, 'Acrobot-v1': 100, 'Pendulum-v1': 100, 'MountainCarContinuous-v0': 109}
thresholds = {'CartPole-v0': 195, 'MountainCar-v0': -150, 'Acrobot-v1': -150, 'Pendulum-v1': -150, 'MountainCarContinuous-v0': 90}
max_steps = {'CartPole-v0': 200, 'MountainCar-v0': 200, 'Acrobot-v1': 200, 'Pendulum-v1': 200, 'MountainCarContinuous-v0': 999}
dts = {'CartPole-v0': 0.02, 'MountainCar-v0': 1, 'Acrobot-v1': 0.2, 'Pendulum-v1': 0.05, 'MountainCarContinuous-v0': 1}

class Environment:
    # Heavily based on:
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

    def __init__(self, name, seed=0):
        if name not in envs:
            print(f'{name} not known. Choose from {str(envs)}')
            return

        self.env = gym.make(name)
        self.env.seed(seed)
        self.name = name

        self.min_episodes_val = min_episodes[name]
        self.threshold_val = thresholds[name]
        self.max_steps_val = max_steps[name]
        self.dt_val = dts[name]

    def step(self, action:np.ndarray, getInfo=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, info = self.env.step(action)
        return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))
    
    def tf_step(self, action) -> List[tf.Tensor]:
        return tf.numpy_function(self.step, [action], [tf.float32, tf.int32, tf.int32])
    
    def get_new_starting_state(self):
        return tf.constant(self.env.reset(), dtype=tf.float32)
    
    def get_num_actions(self):
        try:
            return self.env.action_space.n
        except AttributeError as e:
            return self.env.action_space._shape[0]

    def min_episodes(self):
        return self.min_episodes_val
    
    def threshold(self):
        return self.threshold_val
    
    def max_steps(self):
        return self.max_steps_val
    
    def dt(self):
        return self.dt_val
    
    def get_name(self):
        return self.name
