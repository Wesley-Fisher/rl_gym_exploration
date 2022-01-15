from typing import Tuple

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
        return "SimpleAC"

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


class PIDModel(GymExplorationModel):
    def __init__(self, i, p, integ, d, goal, eps=-1):
        super().__init__()
        self.PID = None
        self.i = i
        self.p = p
        self.integ = integ
        self.d = d
        self.goal = goal
        self.eps = eps
    
    def get_name(self):
        return "PID"

    def new_episode(self):
        self.pid = PID(self.p, self.integ, self.d, setpoint=0)
        self.pid.sample_time = 0.02
    
    def call(self, state: tf.Tensor):
        state = tf.expand_dims(state, 0)
  
        c = self.pid(self.goal - state[0,0,self.i])
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

'''
print("Actor-Critic Demo")
env = Environment('MountainCar-v0')
model = SimpleActorCriticModel(env)
animator = Animator('Demo', env, model, '1000')
director = Director(env, model, animator)
director.train(1000)
'''


print("PID Demo")
env = Environment('MountainCar-v0')
model = PIDModel(0, 0.0001, 0, 10, 0.5, 0.0001)
animator = Animator('Demo', env, model, '100')
director = Director(env, model, animator)
director.train(100)
