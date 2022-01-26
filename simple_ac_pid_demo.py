from typing import Tuple
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras import layers

from simple_pid import PID
from src.animator import Animator

from src.environment import Environment
from src.director import Director
from src.model import GymExplorationModel
from src import util


class SimpleDiscreteActorCriticModel(GymExplorationModel):
    # Heavily based on:
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

    def __init__(self, env):
        super().__init__()

        self.common = layers.Dense(128, activation="relu")
        self.actor = layers.Dense(env.get_num_actions())
        self.critic = layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.gamma = 0.99

        # Values for Training
        self.action_probs = None
        self.values = None
        self.rewards = None

    def get_name(self):
        return "SimpleACDiscrete"

    def new_episode(self):
        self.action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    @tf.function
    def call_inner(self, state: tf.Tensor):
        print("Inner Model Function (tf function)")
        x = self.common(state)
        action_logits = self.actor(x)
        value = self.critic(x)
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

        return action

    def post_step(self, state: tf.Tensor,
                        reward: tf.Tensor,
                        done: bool):
        i = self.rewards.size()
        self.rewards = self.rewards.write(i, reward)

    def post_episode_train(self, tape):
        if self.action_probs.size() == 0:
            return
        action_probs = self.action_probs.stack()
        values = self.values.stack()
        rewards = self.rewards.stack()

        # Get returns
        returns = util.get_expected_return(rewards, self.gamma, self.rewards.size())

        # Reshape, and calculate loss for Actor-Critic
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 
        loss = self.compute_ActorCritic_loss(action_probs, values, returns)

        # Update Model
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def compute_ActorCritic_loss(self,
                     action_probs: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss


class SimpleContinuousActorCriticModel(GymExplorationModel):
    # Heavily based on:
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb
    # https://towardsdatascience.com/a-minimal-working-example-for-continuous-policy-gradients-in-tensorflow-2-0-d3413ec38c6b

    def __init__(self, env, scale, alpha, eps, critic_scale):
        super().__init__()
        self.env = env
        self.scale = scale
        self.alpha = alpha
        self.eps = eps
        self.critic_scale = critic_scale

        self.common1 = layers.Dense(32, activation="relu")
        self.common2 = layers.Dense(32, activation="relu")
        self.actor = layers.Dense(4, activation="relu")
        self.actor_mu = layers.Dense(1, activation='tanh')
        self.actor_scale = layers.Multiply()
        self.actor_sig = layers.Dense(1, activation='sigmoid')
        self.critic = layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        self.gamma = 0.99

        # Values for Training
        self.states = None
        self.actions = None
        self.values = None
        self.rewards = None

    def get_name(self):
        return "SimpleACContinuous"

    def new_episode(self):
        self.actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self.rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        self.states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    @tf.function
    def call_inner(self, state: tf.Tensor):
        print("Inner Model Function (tf function)")
        x = self.common1(state)
        x = self.common2(x)
        act = self.actor(x)
        mu = self.actor_mu(act) * self.scale
        sigma = self.actor_sig(act)
        value = self.critic(x) * self.critic_scale
        return mu, sigma, value

    
    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Prep for recording data
        i = self.rewards.size()

        # Calculations
        action, sig, value = self.call_inner(state)
        action = action + random.gauss(0.0, self.eps)

        # Handle Action
        self.actions = self.actions.write(i, action)

        # Handle Value
        self.values = self.values.write(i, tf.squeeze(value))

        # Store State
        self.states = self.states.write(i, tf.squeeze(state))

        return action

    def post_step(self, state: tf.Tensor,
                        reward: tf.Tensor,
                        done: bool):
        i = self.rewards.size()
        self.rewards = self.rewards.write(i, reward)

    def post_episode_train(self, tape):
        if self.actions.size() == 0:
            return
        actions = self.actions.stack()
        values = self.values.stack()
        rewards = self.rewards.stack()
        states = self.states.stack()

        # Get returns
        returns = util.get_expected_return(rewards, self.gamma, self.rewards.size())

        # Get Action PDF values
        action_pdfs = self.calculate_action_pdfs(states, actions, self.states.size())

        # Reshape, and calculate loss for Actor-Critic
        action_pdfs, values, returns = [tf.expand_dims(x, 1) for x in [action_pdfs, values, returns]] 
        loss = self.compute_ActorCritic_loss(action_pdfs, values, returns)

        # Update Model
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    def calculate_action_pdfs(self, states, actions, n):
        pdfs = tf.TensorArray(dtype=tf.float32, size=n)

        for i in tf.range(n):
            state = states[i]
            state = tf.reshape(state, [1, self.env.env.observation_space.shape[0]])
            act = actions[i]

            # Run network
            mu, sigma, value = self.call_inner(state)

            # Gaussian PDF
            expo_root = (act - mu) / sigma
            den = sigma * tf.sqrt(2*np.pi)
            pdf = tf.exp(-0.5 * expo_root ** 2) / den
            
            pdfs = pdfs.write(i, pdf)
        pdfs = pdfs.stack()
        return pdfs

    def compute_ActorCritic_loss(self,
                     action_pdfs: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:
        advantage = returns - values

        action_log_pdfs = tf.math.log(action_pdfs + 1e-5)
        actor_loss = -tf.math.reduce_sum(action_log_pdfs * advantage)

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
model = SimpleDiscreteActorCriticModel(env)
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator)
director.train(100)


print("CartPole-v0 PID Demo")
env = Environment('CartPole-v0')
model = PIDModel(idx=2, kp=-1.0, ki=0, kd=0, goal=0, eps=-1, continuous=False)
animator = Animator('Demo', env, model, '5', 1, show_live=True)
director = Director(env, model, animator)
director.train(5)



print("MountainCar-v0 Actor-Critic Demo")
env = Environment('MountainCar-v0')
model = SimpleDiscreteActorCriticModel(env)
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator)
director.train(100)


print("MountainCar-v0 PID Demo")
env = Environment('MountainCar-v0')
model = PIDModel(idx=0, kp=1.0, ki=1, kd=0, goal=0.5, eps=0.001, continuous=False)
animator = Animator('Demo', env, model, '5', 1)
director = Director(env, model, animator)
director.train(5)



print("Acrobot-v1 Actor-Critic Demo")
env = Environment('Acrobot-v1')
model = SimpleDiscreteActorCriticModel(env)
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator)
director.train(100)



print("Acrobot-v1 PID Demo")
env = Environment('Acrobot-v1')
model = PIDModel(idx=0, kp=0, ki=0, kd=1, goal=-1, eps=0.0001, continuous=False)
animator = Animator('Demo', env, model, '5', 1)
director = Director(env, model, animator)
director.train(5)



print("Pendulum-v1 Actor-Critic Demo")
env = Environment('Pendulum-v1')
model = SimpleContinuousActorCriticModel(env, 2.0, 0.0001, 0.05, 1000)
animator = Animator('Demo', env, model, '100', 20)
director = Director(env, model, animator)
director.train(300)