from copy import copy

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from memory import Memory

np.random.seed(420)


def empty_state():
	# return np.array(
	# 	[[[0 for r in range(3)] for c in range(3)] for b in range(3)]
	# )
	return np.zeros((3, 3, 3))


def get_available_actions(state):
	# Helper function to get the available actions for any state
	# available = []
	# for i in range(3):
	# 	for j in range(3):
	# 		for k in range(3):
	# 			if state[i][j][k] == 0:
	# 				available.append(f'{i}{j}{k}')
	# return available
	return np.argwhere(state == 0)


def flatten_state(state):
	# return [c for b in state for r in b for c in r]
	return state.flatten().reshape(1, 27)


def get_action_from_idx(idx):
	return idx // 9, (idx % 9) // 3, idx % 3


def get_action_string(action, mark):
	return mark + ''.join(str(x) for x in action)


class DQNAgent(object):
	def __init__(
			self, mark, memory_size=1000, batch_size=1000,
			alpha=0.001, gamma=0.99, epsilon=0.2,
	):
		self.mark = mark
		self.model = self.create_model(alpha)
		self.target = copy(self.model)

		self.memory = Memory(memory_size)
		self.memory_size = memory_size
		self.batch_size = batch_size

		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

	# Function signature as required by the driver program
	def act(self, ava_actions, state):
		# Greedily pick the best playable move and return it
		selected_move = self.greedy(state)
		return get_action_string(selected_move, self.mark)

	def greedy(self, state):
		# Compute Q values for this state
		q_values = self.model.predict(flatten_state(state))

		# Greedily select the highest Q value move
		move_idx = int(np.argmax(q_values[0]))
		move = get_action_from_idx(move_idx)

		# If the move is invalid,
		if state[move] != 0:
			# Iterate over moves in order of Q value until we find a valid one
			all_moves = np.argsort(-q_values[0])
			for move_idx in all_moves[1:]:
				move = get_action_from_idx(move_idx)
				if state[move] == 0:
					break

		# Return selected move
		return move

	def epsilon_greedy(self, state, epsilon=None):
		epsilon = epsilon if epsilon is not None else self.epsilon

		# epsilon denotes the probability of exploration
		if np.random.sample() <= epsilon:
			# Return a random playable move
			available_actions = get_available_actions(state)
			idx = np.random.choice(len(available_actions))
			return available_actions[idx]

		# 1 - epsilon gives the probability of exploitation
		# Return the best playable move
		return self.greedy(state)

	def load_model(self, model_path=None):
		model_path = model_path if model_path else 'models/model1'
		self.model = tf.keras.models.load_model(model_path)

	def save_model(self, model_path=None):
		model_path = model_path if model_path else 'models/model1'
		self.model.save(model_path)

	def train(
			self, opponent, epochs=1000, batch_size=None,
			alpha=None, gamma=None, epsilon=None,
	):
		alpha = alpha if alpha is not None else self.alpha
		gamma = gamma if gamma is not None else self.gamma
		epsilon = epsilon if epsilon is not None else self.epsilon
		batch_size = batch_size if batch_size is not None else self.batch_size

		# TODO: Play games against opponent using epsilon-greedy strategy,
		#  saving experiences in Experience Replay memory
		# TODO: After 'x' experience steps, sample training batch from memory
		# TODO: Use target network to generate target Q values for training
		# TODO: Train network using generated target Q values
		# TODO: Update target network by copying weights from learning network
		# TODO: Every 'y' epochs, play 100 games using greedy strategy and
		#  plot win rate for observation.
		# TODO: Save model after training
		return

	def create_model(self, alpha=None):
		alpha = alpha if alpha is not None else self.alpha

		# Use HeUniform initializer to initialize model weights
		init = HeUniform()

		model = Sequential()
		model.add(Dense(
			24, input_shape=(27,), activation='relu', kernel_initializer=init
		))
		model.add(Dense(12, activation='relu', kernel_initializer=init))
		model.add(Dense(27, activation='linear', kernel_initializer=init))
		model.compile(loss=Huber, optimizer=Adam(alpha), metrics=['accuracy'])
		return model


def main():
	# Training Hyperparameters
	memory_size = 10000  # Experience Replay memory size
	alpha = 0.001  # Model learning rate
	gamma = 0.99  # Reward discount factor
	epsilon = 0.2  # Epsilon for exploration-exploitation
	batch_size = 1000  # Size of sample selected from memory for training

	# Initialize model
	dqn_agent = DQNAgent(
		mark='1', memory_size=memory_size, batch_size=batch_size,
		alpha=alpha, gamma=gamma, epsilon=epsilon,
	)

	# TODO: Check if stored model weights available
	# TODO: Create a RandomAgent or other OpponentAgent to train against
	# TODO: Run training epochs
	# TODO: Measure performance and generate plot
	# TODO: Save updated model weights
	return


if __name__ == "__main__":
	main()
