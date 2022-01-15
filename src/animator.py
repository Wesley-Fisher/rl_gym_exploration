import matplotlib.pyplot as ply

class Animator:
    def __init__(self, env, episode_modulo=10):
        self.env = env

        self.episode_modulo = episode_modulo
        self.episode_count = -1
    
    def start_new_episode(self):
        self.episode_count = self.episode_count + 1
    
    def end_episode(self):
        pass

    def finish(self):
        self.episode_count = -1
    
    def step(self, state, action):
        pass

