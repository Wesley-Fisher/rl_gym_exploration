from typing import Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from simple_pid import PID
from src.animator import Animator

from src.environment import Environment
from src.director import Director
from src.model import GymExplorationModel
from src import util


class SimpleActorCriticModel(GymExplorationModel):
    # Heavily based on:
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

    def __init__(self, env, alpha=0.01, common=[128], actors=[], critics=[], discretization=0, scale=0):
        super().__init__()

        self.common = []
        for c in common:
            self.common.append(layers.Dense(c, activation="relu"))
        

        self.discretization = discretization
        self.scale = scale
        if self.discretization <= 0:
            self.act_N = env.get_num_actions()
        else:
            self.act_N = 2 * self.discretization + 1
        
        self.actor_layers = []
        for a in actors:
            self.actor_layers.append(layers.Dense(a, activation="relu"))
        self.actor = layers.Dense(self.act_N)

        
        self.critic_layers = []
        for c in critics:
            self.critic_layers.append(layers.Dense(c, activation="relu"))
        self.critic = layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.gamma = 0.99

        # Values for Training
        self.action_probs = None
        self.values = None
        self.rewards = None

    def get_name(self):
        return "SimpleAC"

    def new_episode(self):
        self.action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    @tf.function
    def call_inner(self, state: tf.Tensor):
        print("Inner Model Function (tf function)")
        x = state
        for common_layer in self.common:
            x = common_layer(x)

        y = x
        for actor_layer in self.actor_layers:
            y = actor_layer(y)
        action_logits = self.actor(y)
        
        z = x
        for critic_layer in self.critic_layers:
            z = critic_layer(z)
        value = self.critic(z)
        return action_logits, value

    
    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Prep for recording data
        i = self.rewards.size()

        # Calculations
        action_logits, value = self.call_inner(state)

        # Handle Action
        action = tf.random.categorical(action_logits, 1)[0, 0] 
        action_probs = tf.nn.softmax(action_logits)
        self.action_probs = self.action_probs.write(i, action_probs[0, action])

        # Handle Value
        self.values = self.values.write(i, tf.squeeze(value))

        # Continuous / Discretized Action
        if self.discretization > 0:
            action = [self.discretize_action(action)]

        return action

    def discretize_action(self, action):
        action = action - self.discretization
        action = action / self.discretization
        action = action * self.scale
        return action

    def post_step(self, state: tf.Tensor,
                        reward: tf.Tensor,
                        done: bool):
        i = self.rewards.size()
        self.rewards = self.rewards.write(i, reward)

    def post_episode_train(self, tape, scale_return=lambda returns: returns):
        action_probs = self.action_probs.stack()
        values = self.values.stack()
        rewards = self.rewards.stack()

        # Get returns
        returns = util.get_expected_return(rewards, self.gamma, self.rewards.size(), False)
        returns = scale_return(returns)

        # Reshape, and calculate loss for Actor-Critic
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 
        loss = self.compute_ActorCritic_loss(action_probs, values, returns)

        # Update Model
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return float(loss)

    def compute_ActorCritic_loss(self,
                     action_probs: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss


class PIDModel(GymExplorationModel):
    def __init__(self, idx=0, kp=0, ki=0, kd=0, goal=0, eps=-1, continuous=False):
        super().__init__()
        self.PID = None
        self.idx = idx
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.goal = goal
        self.eps = eps
        self.continuous = continuous
    
    def get_name(self):
        return "PID"

    def new_episode(self):
        self.pid = PID(self.kp, self.ki, self.kd, setpoint=0)
        self.pid.sample_time = 0.02
    
    def call(self, state: tf.Tensor):
        state = tf.expand_dims(state, 0)
  
        c = self.pid(-(self.goal - state[0,0,self.idx]))
        if self.continuous:
            return np.array([c])
        act = 0

        if self.eps < 0:
            if c < 0:
                act = 0
            else:
                act = 1
            return act
        
        if c < -self.eps:
            act = 0
        elif c > self.eps:
            act = 2
        else:
            act = 1
        return act


print("CartPole-v0 Actor-Critic Demo")
env = Environment('CartPole-v0')
scale_return = lambda returns: returns / 1.0
model = SimpleActorCriticModel(env, alpha=0.001, common=[64,64], actors=[8], critics=[4])
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator, custom_scale_return=scale_return, scale_state=False)
director.train(1000)


print("MountainCar-v0 Actor-Critic Demo")
env = Environment('MountainCar-v0')
montaincar_reward = lambda state: (tf.math.abs(state[1]*14)**1 + state[0]) * 10
scale_return = lambda returns: returns / 10.0
model = SimpleActorCriticModel(env, alpha=0.001, common=[256,128, 32], critics=[16,4])
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator, custom_reward=montaincar_reward, custom_scale_return=scale_return, scale_state=False)
director.train(1000)


print("Acrobot-v1 Actor-Critic Demo")
env = Environment('Acrobot-v1')
acrobot_reward = lambda state: -state[1] * 10
model = SimpleActorCriticModel(env, common=[64, 64])
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator, custom_reward=acrobot_reward)
director.train(100)


print("Pendulum-v1 Actor-Critic Demo")
env = Environment('Pendulum-v1')
model = SimpleActorCriticModel(env, discretization=5, scale=2)
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator)
director.train(100)


print("MountainCarContinuous-v0 Actor-Critic Demo")
env = Environment('MountainCarContinuous-v0')
model = SimpleActorCriticModel(env, discretization=5, scale=1)
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator)
director.train(100)
