import collections
import tqdm
import statistics

import tensorflow as tf

class Director:
    # Heavily based on
    # https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

    def __init__(self, env, model, animator, custom_reward=lambda s: 0.0):
        self.env = env
        self.model = model
        self.animator = animator
        self.custom_reward = custom_reward
    
    def train(self, max_episodes):
        # Keep last episodes reward
        episodes_reward: collections.deque = collections.deque(maxlen=self.env.min_episodes())

        solved = False
        with tqdm.trange(max_episodes) as t:
            for i in t:
                ep_reward, custom_reward = self.run_training_episode()
                episode_reward = int(ep_reward)
                
                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)
            
                t.set_description(f'Episode {i}')
                t.set_postfix(
                    custom_reward=custom_reward,
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
        custom_reward = 0.0

        for t in tf.range(self.env.max_steps()):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)
        
            action = self.model(state)

            self.animator.step(state, action)

            # Apply action to the environment to get next state and reward
            state, reward, done = self.env.tf_step(action)
            state.set_shape(state_shape)
            reward = tf.cast(reward, dtype=tf.float32)
            custom_reward = self.custom_reward(state)

            self.model.post_step(state, reward + custom_reward, done)

            r = float(reward)
            episode_reward = episode_reward + float(r)

            custom_reward = custom_reward + float(r) + float(custom_reward)

            if tf.cast(done, tf.bool):
                break
        
        # Capture final frame
        self.animator.step(state, action)

        return episode_reward, custom_reward