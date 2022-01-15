import numpy as np

import cv2
import imageio

class Animator:
    def __init__(self, env, episode_modulo=10):
        self.env = env

        self.episode_modulo = episode_modulo
        self.episode_count = -1

        self.dt = env.dt()
        self.period = 1.0 / 30.0 # Aim for 30 fps
        self.tau = self.period + 1.0

        self.images = []
        self.name = "Animation"
    
    def start_new_episode(self):
        self.episode_count = self.episode_count + 1
    
    def end_episode(self):
        pass

    def finish(self):
        self.episode_count = 0

        with imageio.get_writer(self.name+'.gif', mode='I') as writer:
            for idx, frame in enumerate(self.images):
                writer.append_data(frame)
    
    def step(self, state, action):
        # Don't animate all episodes
        if self.episode_modulo < 0:
            return
        if self.episode_modulo != 0 and \
           self.episode_count % self.episode_modulo != 0:
           return
        
        if self.tau > self.period:
            self.animate_frame(state, action)
            self.tau = 0.0
        
        self.tau = self.tau + self.dt

    def animate_frame(self, state, action):
        frame = self.env.env.render(mode='rgb_array')
        frame = np.flip(frame, 2) # OpenCV is BGR
        frame = frame.astype(np.uint8)

        text = f'{self.episode_count+1}'
        frame = cv2.putText(frame, text, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        cv2.imshow(self.name, frame)
        cv2.waitKey(int(self.dt*1000))

        frame = np.flip(frame, 2)
        frame = frame.astype(np.uint8)
        self.images.append(frame)
