import collections
import tqdm
import statistics

import tensorflow as tf

class Director:
    # Heavily based on
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

    def __init__(self, env, model):
        self.env = env
        self.model = model
    
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
            episode_reward = float(self.run_episode())
            self.model.post_episode_train(tape)
        
        return episode_reward

    def run_episode(self):
        state = self.env.get_new_starting_state()
        state_shape = state.shape

        self.model.new_episode()
        episode_reward = 0.0

        for t in tf.range(self.env.max_steps()):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)
        
            action = self.model(state)
        
            # Apply action to the environment to get next state and reward
            state, reward, done = self.env.tf_step(action)
            state.set_shape(state_shape)

            self.model.post_step(state, reward, done)

            r = float(reward)
            episode_reward = episode_reward + float(r)

            if tf.cast(done, tf.bool):
                break
        
        return episode_reward