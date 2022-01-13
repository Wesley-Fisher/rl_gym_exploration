import collections
import tqdm
import statistics

import tensorflow as tf

from . import util

class ActorCriticDirector:
    # Heavily based on
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

    def __init__(self, env, model, gamma=0.99):
        self.env = env
        self.model = model
        self.gamma = gamma
    
    def train(self, max_episodes):
        # Keep last episodes reward
        episodes_reward: collections.deque = collections.deque(maxlen=self.env.min_episodes())

        solved = False
        with tqdm.trange(max_episodes) as t:
            for i in t:
                episode_reward = int(self.run_training_episode())
                
                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)
            
                t.set_description(f'Episode {i}')
                t.set_postfix(
                    episode_reward=episode_reward, running_reward=running_reward)
            
                if running_reward > self.env.threshold() and i >= self.env.min_episodes():
                    solved = True
                    break

        if solved:
            print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
        else:
            print("Not Solved!")
    
    def run_training_episode(self):

        with tf.GradientTape() as tape:
            action_probs, values, rewards, episode_reward = self.run_episode()

            returns = util.get_expected_return(rewards, self.gamma)

            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

            loss = self.model.compute_loss(action_probs, values, returns)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return episode_reward

    def run_episode(self):
        state = self.env.get_new_starting_state()
        state_shape = state.shape

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        for t in tf.range(self.env.max_steps()):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)
        
            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self.model(state)
        
            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])
        
            # Apply action to the environment to get next state and reward
            state, reward, done = self.env.tf_step(action)
            state.set_shape(state_shape)
        
            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        episode_reward = tf.math.reduce_sum(rewards)
        
        return action_probs, values, rewards, episode_reward