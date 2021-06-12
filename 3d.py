from random import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np
from gym_tictactoe.envs.tictactoe_env import \
	TicTacToeEnv, after_action_state, \
	agent_by_mark, check_game_status


class HumanAgent:
	# HumanAgent class, mostly unchanged from the provided template
	def __init__(self, mark):
		self.mark = mark

	def act(self, ava_actions, state):
		# Loop until valid input
		while True:
			# Get user input and check for quit signal
			action = input('Enter position [000 - 222], q for quit: ')
			if action.lower() == 'q':
				return None

			try:
				# Test if input is valid
				if action not in ava_actions:
					raise ValueError()

			# Continue iterating if input invalid, otherwise break and return
			except ValueError:
				print(f"Illegal position: '{action}'")
			else:
				break

		return self.mark + action


def create_model():
	# TODO: Tweak network architecture and hyperparameters
	learning_rate = 0.001
	init = HeUniform()

	model = Sequential()
	model.add(Dense(24, input_shape=(27,), activation='relu', kernel_initializer=init))
	model.add(Dense(12, activation='relu', kernel_initializer=init))
	model.add(Dense(27, activation='linear', kernel_initializer=init))
	model.compile(loss=Huber, optimizer=Adam(learning_rate), metrics=['accuracy'])
	return model


def empty_state():
	# return np.zeros((3, 3, 3))
	return np.array([[[0 for r in range(3)] for c in range(3)] for b in range(3)])


def get_available_actions(state):
	# Helper function to get the available actions for any state
	# We need this because the env doesn't provide a method to generate
	# available actions for new states (without creating a new env object)
	# Further, since we're dealing with 2D, we can avoid generating invalid
	# moves that would need to be filtered later
	# return np.argwhere(state == 0)
	available = []
	# Check if position vacant and add it as a valid move
	for i in range(3):
		for j in range(3):
			for k in range(3):
				if state[i][j][k] == 0:
					available.append(f'{i}{j}{k}')
	return available


def flatten_state(state):
	# return state.flatten()
	return [c for b in state for r in b for c in r]


def get_action_from_idx(idx):
	return idx // 9, (idx % 9) // 3, idx % 3


def get_action_string(action, mark):
	return mark + ''.join(str(x) for x in action)


class AIAgent:
	# Agent that uses Deep Q Learning

	# Array that stores evaluations for positions
	# dp = [None] * 3 ** 9

	def __init__(self, mark, model):
		self.mark = mark
		self.model = model

	def act(self, ava_actions, state):
		# TODO: Forward propagate the network
		# TODO: Find the action with the highest Q value
		# TODO: Format and return chosen action
		# Return the selected best action
		# return self.mark + best_action
		q_values = self.model.predict(flatten_state(state))
		print(q_values)
		print(q_values.shape)
		move_idx = int(np.argmax(q_values[0]))
		move = get_action_from_idx(move_idx)

		# TODO: Create another act function that does explore and exploit
		# next_state, reward = game.play_board(deepcopy(current_state), move)
		# next_state = game.step(move)[0]
		# TODO: Check if if valid position before trying move and
		#  enter the random move loop early
		# if move not in ava_actions:
		# 	return -1
		if state[move] != 0:
			# TODO: Consider using argsort or something and play the next best move
			all_moves = np.argsort(-q_values[0])
			for move_idx in all_moves[1:]:
				move = get_action_from_idx(move_idx)
				if state[move] == 0:
					break

		return get_action_string(move, self.mark)

	@staticmethod
	def get_available_actions(state):
		# Helper function to get the available actions for any state
		# We need this because the env doesn't provide a method to generate
		# available actions for new states (without creating a new env object)
		# Further, since we're dealing with 2D, we can avoid generating invalid
		# moves that would need to be filtered later
		available = []

		for i in range(3):
			for j in range(3):
				for k in range(3):
					# Check if position vacant and add it as a valid move
					if state[i][j][k] == 0:
						available.append(f'{i}{j}{k}')
		return available


def main():
	# Driver code to run 2D human-vs-AI TicTacToe
	# Create environment
	env = TicTacToeEnv()

	# Assign player 1 and 2 randomly to human and agent
	marks = ['1', '2']
	shuffle(marks)
	agents = [HumanAgent(marks[0]), AIAgent(marks[1])]
	print(f'Human: Player {marks[0]}. Machine: Player {marks[1]}')

	# Counter for moves to check if game ended in draw
	moves = 0

	while True:
		# Get the player to move
		agent = agent_by_mark(agents, str(env.show_turn()))

		# Get possible moves for this player and ask for chosen move
		ava_actions = env.available_actions()
		action = agent.act(ava_actions, env._world)

		# Check if human wants to quit
		if action is None:
			print("==== Exiting. ====")
			break

		# Perform the move and render the board
		state, reward, done, info = env.step(action)
		env.render()
		print()

		# If game over, show result and break
		if done:
			env.show_result()
			break

		# Else increment move and check for draw
		moves += 1
		if moves == 9:
			print("==== Finished: Game ended in draw. ====")
			break


if __name__ == '__main__':
	main()
