import gym
import numpy as np

np.random.seed(0)
env = gym.make("MountainCar-v0")


q_table = np.load("qtables/q_table_240000.npy")

d_size = [20] * len(env.observation_space.high)
d_w_size = (env.observation_space.high - env.observation_space.low) / d_size


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / d_w_size
    return tuple(discrete_state.astype(np.int32))


discrete_state = get_discrete_state(env.reset())


done = False
while not done:
    if not done:
        action = np.argmax(q_table[discrete_state])
        new_State, reward, done, _ = env.step(action)
        env.render()

    discrete_state = get_discrete_state(new_State)


env.close()
