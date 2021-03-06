import tensorflow as tf

class GymExplorationModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def get_name(self):
        raise NotImplementedError

    def new_episode(self):
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
    
    def post_episode_train(self, tape, scale_return=lambda returns: returns):
        # If relevant do training
        # Tensorflow tf.GradientTape available
        return
