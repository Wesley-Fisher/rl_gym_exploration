from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


class SimpleActorCriticModel(tf.keras.Model):
    # Heavily based on:
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

    def __init__(self, env):
        super().__init__()

        self.common = layers.Dense(128, activation="relu")
        self.actor = layers.Dense(env.get_num_actions())
        self.critic = layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    
    def call(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(state)
        return self.actor(x), self.critic(x)

    def compute_loss(self,
                     action_probs: tf.Tensor,
                     values: tf.Tensor,
                     returns: tf.Tensor) -> tf.Tensor:

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss
