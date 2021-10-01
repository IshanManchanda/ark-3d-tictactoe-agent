from copy import copy

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from memory import Memory
from utils import flatten_state, get_action_from_idx, get_action_string, \
	get_available_actions


class DQNAgent(object):
	def __init__(
			self, mark, memory_size=1000, batch_size=1000,
			alpha=0.001, gamma=0.99,
			eps_start=0.2, eps_decay=1.0, eps_min=0.02
	):
		self.mark = mark
		self.model = self.create_model(alpha)
		self.target = copy(self.model)

		self.memory = Memory(memory_size)
		self.memory_size = memory_size
		self.batch_size = batch_size

		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = eps_start
		self.eps_decay = eps_decay
		self.eps_min = eps_min

	def set_mark(self, mark):
		# Function to update mark for training
		self.mark = mark

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

	def train_game(self, opponent):
		# TODO: Play a game from start to end
		pass

	def train(self, opponent, epochs=1000):
		# Outer training loop, continue until converged/epochs over
		for _ in range(epochs):
			# Inner loop, play certain number of games
			for _ in range(100):
				# TODO: Play games against opponent using epsilon-greedy strategy,
				#  saving experiences in Experience Replay memory
				self.train_game(opponent)

			# TODO: After 'x' experience steps, sample training batch from memory
			# TODO: Use target network to generate target Q values for training
			# TODO: Train network using generated target Q values
			# TODO: Update target network by copying weights from learning network
			# TODO: Every 'y' epochs, play 100 games using greedy strategy and
			#  plot win rate for observation.
			# TODO: Save model after training
			pass

		return

	@staticmethod
	def create_model(alpha):
		# Use HeUniform initializer to initialize model weights
		init = HeUniform()

		model = Sequential()
		model.add(Dense(
			128, input_shape=(27,), activation='relu', kernel_initializer=init
		))
		# model.add(Dense(128, activation='relu', kernel_initializer=init))
		model.add(Dense(27, activation='linear', kernel_initializer=init))
		model.compile(loss=Huber, optimizer=Adam(alpha), metrics=['accuracy'])
		return model
