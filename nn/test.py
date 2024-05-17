import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from a2c_agent import A2CAgent


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (
        cumulative_sum[window_size:] - cumulative_sum[:-window_size]
    ) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[: window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# Update Q' with Q weights
def update_target(Q, target):
    for i in range(Q.L):
        W1, b1 = Q.NN[i]
        W2, b2 = target[i]
        W2[:] = W1[:]
        b2[:] = b1[:]


def train_DQN(agent, env, num_episodes, model_id="ac-5"):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    k = 0
    for i in range(num_episodes):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            episode_return = 0
            obs, _ = env.reset()
            done = False
            for i_episode in range(int(num_episodes / 10)):
                state = obs.reshape(-1, 1)
                # while not done:
                action = agent.act(state, model_id)
                max_q_value = (
                    agent.max_q_value(state, model_id) * 0.005 + max_q_value * 0.995
                )  # 平滑处理
                max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值

                obs, reward, done, info, _ = env.step(action)
                next_state = obs.reshape(-1, 1)
                # replay_buffer.add(state, action, reward, next_state, done)
                # agent.learn(state, next_state, action, reward, model_id, done=done,loss_function="MSE")
                # state = next_stat

                # sprint(yhat,"yhat")
                # agent.train(state[True,:], yhat,epoch=1)

                # state = next_state
                episode_return += reward
                k += 1
                if done:
                    break

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    return return_list, max_q_value_list


import gym
import numpy as np
from optimx import make


env = gym.make("CartPole-v1")


import numpy as np
from collections import deque

num_episodes = 10000

# Global variables
NUM_EPISODES = 1000
MAX_TIMESTEPS = 1000
AVERAGE_REWARD_TO_SOLVE = 1950
NUM_EPS_TO_SOLVE = 100
NUM_RUNS = 20
GAMMA = 0.95
EPSILON_DECAY = 0.997
update_size = 10
hidden_layer_size = 24
num_hidden_layers = 2
action_list = [a for a in range(env.action_space.n)]

model_name = f"new_model"
VERSION = "0.0"
MODEL_ENV = "dev"
model_db2 = make(
    f"cache/{model_name}-v{VERSION}",
    db_name="dqnmodel_test22.db",
    env=MODEL_ENV,
    db_type="diskcache",
)

# from nn import sigmoid
from layers import Input, Dense

_layers = [
    Input(4),
    Dense(units=32, activation="tanh"),
    Dense(units=16, activation="tanh"),
    Dense(units=2, activation="sig"),
]
model = A2CAgent(layers=_layers, model_db=model_db2)

return_list, max_q_value_list = train_DQN(
    model, env, num_episodes, model_id="new_linear-shima-2"
)

env_name = "CartPole-v1"
episodes_list = list(range(len(return_list)))
mv_return = moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("DQN on {}".format(env_name))
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c="orange", ls="--")
plt.axhline(10, c="red", ls="--")
plt.xlabel("Frames")
plt.ylabel("Q value")
plt.title("DQN on {}".format(env_name))
plt.show()
