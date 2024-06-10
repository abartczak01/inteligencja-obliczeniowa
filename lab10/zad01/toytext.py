import gym
env = gym.make("Blackjack-v1", render_mode="human")

print(env.observation_space)
print(env.action_space)
observation, info = env.reset(seed=42)

for _ in range(60):
   action = env.action_space.sample()
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()

# Toy Text

# stan gry (observation space): dyskretny
# zestaw akcji (action space): dyskretny

# Tuple(Discrete(32), Discrete(11), Discrete(2))
# Discrete(2)