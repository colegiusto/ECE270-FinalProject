import numpy as np
import pyspiel
import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

# Environment
env = rl_environment.Environment("pentago")
num_actions = env.action_spec()["num_actions"]
state_size = env.observation_spec()["info_state"][0]

# Create two DQN agents for self-play
agents = [
    dqn.DQN(
        player_id=i,
        state_representation_size=state_size,
        num_actions=num_actions,
        hidden_layers_sizes=[256, 256],
        replay_buffer_capacity=10_000,
        batch_size=128,
        learning_rate=1e-3,
        update_target_network_every=500,
        learn_every=10,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_duration=50_000,
    )
    for i in range(2)
]

# Training loop
num_episodes = 20_000

for ep in range(num_episodes):
    time_step = env.reset()

    while not time_step.last():
        current_player = time_step.observations["current_player"]
        agent_output = agents[current_player].step(time_step)
        time_step = env.step([agent_output.action])

    # Let both agents observe the terminal state
    for agent in agents:
        agent.step(time_step)

    # Logging
    if ep % 500 == 0:
        print(f"Episode {ep}/{num_episodes} | "
              f"P0 loss: {agents[0].loss} | "
              f"P1 loss: {agents[1].loss} | ")

# Save the trained agents
torch.save(agents[0]._q_network.state_dict(), "agent0.pt")
torch.save(agents[1]._q_network.state_dict(), "agent1.pt")
print("Training complete!")