import gym
import numpy as np

np.random.seed(0)

env = gym.make("MountainCar-v0")

lr = 0.1
d = 0.95
EPISODES = 250000
STATS_EVERY = 5000
epsilon = 0.5

START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)

d_size = [20] * len(env.observation_space.high)
d_w_size = (env.observation_space.high - env.observation_space.low) / d_size

q_table = np.random.uniform(low=-2, high=0, size=(d_size + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / d_w_size
    return tuple(discrete_state.astype(np.int32))


ep_rewards = []
aggr_ep_rewards: dict = {"ep": [], "avg": [], "max": [], "min": []}


for episode in range(EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # action = np.argmax(q_table[discrete_state])
        new_State, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_State)
        # env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - lr) * current_q + lr * (reward + d * max_future_q)
            q_table[discrete_state + (action,)] = new_q

            # print(f"Updating q on ep {episode}")
        elif new_State[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print("We made it on episode {}".format(episode))

        discrete_state = new_discrete_state

    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        epsilon -= epsilon_decay

    if not episode % 10000:
        np.save(f"qtables/q_table_{episode}.npy", q_table)

    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY

        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["max"].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards["min"].append(min(ep_rewards[-STATS_EVERY:]))
        print(
            f"Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}"
        )

env.close()
