import gymnasium as gym


# env = gym.make("CartPole-v1", render_mode="human")

# observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()

env = gym.make("LunarLander-v2",render_mode="human")
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, done, _ = env.step(action)

    if done:
        observation = env.reset()

env.close()