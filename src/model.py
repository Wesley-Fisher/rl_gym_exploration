from re import I
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

from . import util

class GymExplorationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def new_episode():
        # Reset anything needed for a new episode
        raise NotImplementedError
    
    def call(self, state: tf.Tensor):
        # Implement logic to take an action
        raise NotImplementedError

    def post_step(self, state: tf.Tensor,
                        reward: tf.Tensor,
                        done: bool):
        # If relevant, handle data from taking the action
        return
    
    def post_episode_train(self, tape):
        # If relevant do training
        # Tensorflow tf.GradientTape available
        return

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
