import numpy as np
import tensorflow as tf

from agent_dqn import DQNAgent
from agent_random import RandomAgent

# Set random seeds for reproducibility
np.random.seed(1)
tf.random.set_seed(1)


# REVIEW: Consider making this a jupyter nb
def main():
	# Training Hyperparameters
	memory_size = 10000  # Experience Replay memory size
	batch_size = 32  # Size of sample selected from memory for training
	alpha = 1e-3  # Model learning rate
	gamma = 0.99  # Reward discount factor
	# Epsilon parameters for exploration-exploitation
	eps_start = 1.0
	eps_decay = 0.999985
	eps_min = 0.02

	# sync_time: Parameter that determines when to sync target network

	# Initialize model
	dqn_agent = DQNAgent(
		mark='1', memory_size=memory_size, batch_size=batch_size,
		alpha=alpha, gamma=gamma,
		eps_start=eps_start, eps_decay=eps_decay, eps_min=eps_min
	)

	# TODO: Check if stored model weights available

	# Create a RandomAgent to train against
	random_agent = RandomAgent(mark='2')
	# REVIEW: Pass list of opponent agents to train against?
	#  Eventually pass partially trained network for self-play?

	# TODO: Run training epochs
	# dqn_agent.train(RandomAgent)

	# TODO: Measure performance and generate plot
	# TODO: Save updated model weights
	return


if __name__ == "__main__":
	main()
