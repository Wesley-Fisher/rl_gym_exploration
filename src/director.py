import collections
import tqdm
import statistics
import numpy as np

import tensorflow as tf

import sklearn
import sklearn.preprocessing

class Director:
    # Heavily based on
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

    def __init__(self, env, model, animator, custom_reward=lambda s: 0.0):
        self.env = env
        self.model = model
        self.animator = animator
        self.custom_reward = custom_reward
        self.custom_scale_return = custom_scale_return

        # https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
        self.scaler = sklearn.preprocessing.StandardScaler()
        state_samples = np.array([env.env.observation_space.sample() for x in range(10000)])
        self.scaler.fit(state_samples)
    
    def train(self, max_episodes):
        # Keep last episodes reward
        episodes_reward: collections.deque = collections.deque(maxlen=25)
        episodes_custom: collections.deque = collections.deque(maxlen=25)

        solved = False
        with tqdm.trange(max_episodes) as t:
            for i in t:
                ep_reward, custom_reward = self.run_training_episode()
                episode_reward = int(ep_reward)
                
                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                episodes_custom.append(custom_reward)
                running_custom = statistics.mean(episodes_custom)
            
                t.set_description(f'Episode {i}')
                t.set_postfix(
                    custom_reward=custom_reward,
                    running_custom=running_custom,
                    episode_reward=episode_reward,
                    running_reward=running_reward)
            
                if running_reward > self.env.threshold() and i >= self.env.min_episodes():
                    solved = True
                    break

        if solved:
            print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
        else:
            print("Not Solved!")
        
        self.animator.finish()
    
    def run_training_episode(self):
        self.animator.start_new_episode()

        with tf.GradientTape() as tape:
            episode_reward, custom_reward = self.run_episode()
            self.model.post_episode_train(tape)
        
        self.animator.end_episode()
        return float(episode_reward), float(custom_reward)

    def run_episode(self):
        state = self.env.get_new_starting_state()
        state_shape = state.shape

        self.model.new_episode()
        episode_reward = 0.0
        episode_custom_reward = 0.0

        for t in tf.range(self.env.max_steps()):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            scaled_state = self.scale_state(state)            
            action = self.model(scaled_state)

            self.animator.step(state, action)

            # Apply action to the environment to get next state and reward
            state, reward, done = self.env.tf_step(action)
            state.set_shape(state_shape)
            reward = tf.cast(reward, dtype=tf.float32)
            custom_reward = self.custom_reward(state)

            self.model.post_step(state, reward + custom_reward, done)

            r = float(reward)
            episode_reward = episode_reward + float(r)

            episode_custom_reward = episode_custom_reward + float(r) + float(custom_reward)

            if tf.cast(done, tf.bool):
                break
        
        # Capture final frame
        self.animator.step(state, action)

        return episode_reward, episode_custom_reward
    
    def scale_state(self, state):
        return self.scaler.transform(state)