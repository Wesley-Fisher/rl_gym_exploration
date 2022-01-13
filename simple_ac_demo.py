import collections
import tqdm

from src.environment import Environment
from src.director import ActorCriticDirector
from src.model import SimpleActorCriticModel

env = Environment('CartPole-v0')
model = SimpleActorCriticModel(env)

director = ActorCriticDirector(env, model)

director.train(1000)