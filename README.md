# rl_gym_exploration
A repo to explore some OpenAI Gym Environments with reinforcement learning and other controllers.

Originally created based off of tensorflow example code from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb

See: https://gym.openai.com/

# Code Documentation

See full examples of usage in simple_ac_pid_demo.py.


## Animator

A tool for animating episodes, and saving a `.gif` file of training.

`Animator(prefix, env, model, suffix, episode_modulo=10, show_live=False)`

* prefix: a string prefix for the gif file.
* env: the `Environment` in use
* model: the `Model` in use
* suffix: a string suffix for the file (not including `.gif`)
* episode_module: an episode will be animated if `episode_modulo == 0` or `episode_num % episode_modulo == 0`
* show_live: determines whether the OpenCV window is animated live

The file output will be `<prefix>_<env-name>_<model-name>_suffix.gif`


## Director

A class to manage running episodes of a model for training, and coordinate interactions between the model, environment, and animator.

`Director(env, model, animator, custom_reward, custom_scale_return, scale_state=True)`

* env: the `Environment` in use
* model: the `Model` in use
* animator: the `Animator` in use
* custom_reward: a `lambda` function which takes in a state, and returns a custom reward. This is added to the environment reward. Defaults to `lambda s: 0.0`.
* custom_scale_return: a `lambda` function which takes in the returns of an episode, and scales them. This is passed to the `Model` in the `post_episode_train` function. Defaults to `lambda returns: returns`.
* scale_state: if `False`, the raw state is given to the `GymExplorationModel`. If `True`, a random sampling of 10000 states is used with a `sklearn` `StandardScaler`, and a scaled state is given to the model


## Environment Class

Provides access to an OpenAI Gym environment.

`Environment(name, seed=0)`

* name: name of the environment. Currently must be one of `CartPole-v0`, `MountainCar-v0`, `Acrobot-v1`, `Pendulum-v1`, `MountainCarContinuous-v0`
* seed: a random seed


## GymExplorationModel (Base Class)

A base class to act as an agent in the environment. Subclasses `tf.keras.Model`.

### Required Functions

`get_name`

* Returns: name of the model

`call(state)`

* state: the sensed state of the environment (perhaps scaled)
* Returns: the action to be taken
  * See `SimpleActorCriticModel` for a continuous vs discrete action implementation
* Note: may record information for later use

`post_step(state, reward, done)`

* state: resulting state of the environment after applying the action returned by `call`
* reward: the reward received from taking the action returned by `call`, including any custom reward
* done: indicator if the episode is now down


`post_episode_train(tape, scale_return)`

* tape: a `tf.GradientTape` object. If a value is calculated in the function, then `tape.gradient(<variable>, self.trainable_variables)` can be called to calculate the gradients of the value with respect to model variables
* scale_return: custom `lamda` function to scale returns, provided by a `Director`. Takes a set of returns as a Tensor as an object, and returns a scaled version
* Note: intended as a function to trigger any machine learning abilities of a model after each episode is finished


### SimpleActorCriticModel Implementation of GymExplorationModel

An implementation of an Actor-Critic reinforcement learning model. Documentation on sources/inspiration is available in the code.

`SimpleActorCriticModel(num_act=0, alpha=0.01, common=[128], actors=[], critics=[], discretization=0, scale=0)`

* env: the `Environment` in use. Only used to determine the number of actions to be taken
* alpha: the learning rate parameter
* common: a list of integers representing the number of neurons in common layers of the network.
* actors: a list of integers representing the number of neurons in layers of the network specific to the actor. The final layer corresponding to the number of action outputs is added beyond this.
* critics: a list of integers representing the number of neurons in layers of the network specific to the critic. The final layer corresponding to the value output is added beyond this.
* discretization: the number of discretized output value bins to use if the action space is continuous. See below
* scale: the scale to use with discretized outputs. See below

The network begins with zero or more common layers, which perform calculations on the input state. The output of this is then input to layers specific to the Actor and Critic networks, and finally to final layers of each which form the correct final output for the given configuration.

For discrete action spaces, a final layer is added to the Actor portion of the network, with `Dense` neurons. An action pulled from this as a distribution.

For continuous action spaces, the action output is discretized, and the above process is followed. The action at index `0` represents the continuous value `-<scale>`. The action at index `<discretization>` represents the continuous value `0.0`. The action at index `2 * <discretization> + 1` represents the continuous value `+ <scale>`. The outputs of other indices are scaled linearly between these values.
