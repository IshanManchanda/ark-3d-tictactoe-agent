import numpy as np

import matplotlib

import matplotlib.pyplot as plt
from copy import deepcopy

from random import choice
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
# from keras import backend as K

from memory import ExperienceReplay

# from agents import TDAgent
# from games.board import empty_state, is_game_over, flatten_state, open_spots, print_board

from gym_tictactoe.envs.tictactoe_env import \
	TicTacToeEnv, after_action_state, \
	agent_by_mark, check_game_status

np.random.seed(1337)  # for reproducibility
matplotlib.use('Agg')


def empty_state():
	# return np.array([[[0 for r in range(3)] for c in range(3)] for b in range(3)])
	return np.zeros((3, 3, 3))


def get_available_actions(state):
	# Helper function to get the available actions for any state
	# We need this because the env doesn't provide a method to generate
	# available actions for new states (without creating a new env object)
	# Further, since we're dealing with 2D, we can avoid generating invalid
	# moves that would need to be filtered later
	# available = []
	# Check if position vacant and add it as a valid move
	# for i in range(3):
	# 	for j in range(3):
	# 		for k in range(3):
	# 			if state[i][j][k] == 0:
	# 				available.append(f'{i}{j}{k}')
	# return available
	return np.argwhere(state == 0)


def flatten_state(state):
	# return [c for b in state for r in b for c in r]
	return state.flatten()


def get_action_from_idx(idx):
	return idx // 9, (idx % 9) // 3, idx % 3


def get_action_string(action, mark):
	return mark + ''.join(str(x) for x in action)


class DQNAgent(object):
	def __init__(self, model, mark, memory=None, memory_size=1000):
		self.model = model
		self.mark = mark

		if memory is not None:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)

	''', epsilon_rate=0.5'''

	def train_network(
			self, game: TicTacToeEnv, num_epochs=1000, batch_size=50, gamma=0.9,
			epsilon=(.1, 1), epsilon_rate=0.5, reset_memory=False, observe=2):

		model = self.model
		# game.reset_agent()
		game.reset()
		nb_actions = model.output_shape[-1]
		win_count, loss_count, total_reward, total_q = 0, 0, 0.0, 0.0
		batch_probs, avg_reward, avg_q = [], [], []
		total_q, total_reward = 0.0, 0.0
		delta = (epsilon[1] - epsilon[0]) / (num_epochs * epsilon_rate)
		epsilon_set = np.arange(epsilon[0], epsilon[1], delta)

		env = TicTacToeEnv()

		for epoch in range(1, num_epochs + 1):

			loss, winner = 0., None
			num_plies, game_q, game_reward = 0, 0.0, 0.0
			env.reset()
			game_over = False
			# self.clear_frames()

			if reset_memory:
				self.memory.reset_memory()

			if epoch % (num_epochs / 1000) == 0:
				batch_probs.append(self.measure_performance(game, 100))

			while not game_over:

				# pdb.set_trace()
				if np.random.random() > epsilon_set[int(epoch * epsilon_rate - 0.5)]:  # or epoch < observe:
					empty_cells = get_available_actions(current_state)
					move = choice(empty_cells)  # choose move randomly from available moves

				# choose the action for which Q function gives max value
				else:
					q = model.predict(flatten_state(current_state))
					move = int(np.argmax(q[0]))
					game_q += np.amax(q[0])

				next_state, reward = game.play_board(deepcopy(current_state), move)

				game_reward += reward
				num_plies += 1

				# check who, if anyone, has won
				if reward != 0 or len(get_available_actions(next_state)) == 0:
					game_over = True

				'''reward,'''
				transition = [
					flatten_state(current_state), move, reward,
					flatten_state(next_state), game_over
				]
				self.memory.remember(*transition)
				current_state = next_state  # update board state

				if epoch % observe == 0:

					batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
					if batch:
						inputs, targets = batch
						loss += float(model.train_on_batch(inputs, targets))

				'''if checkpoint and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == num_epochs):
					model.save_weights('weights.dat')'''

			if reward == -1 * game.symbol:  # ttt agent's symbol is inverted to get the model's symbol
				win_count += 1

			total_q += game_q / num_plies
			total_reward += game_reward / num_plies
			avg_q.append(total_q / epoch)
			avg_reward.append(total_reward / epoch)
			print("Epoch {:03d}/{:03d} | Loss {:.4f} | Average Q {:.2f} | Average Reward {:.2f} | Wins {}".format(epoch, num_epochs, loss, avg_q[epoch - 1], avg_reward[epoch - 1], win_count))

		game_epochs = [i for i in range(1000)]
		win_probs = [probs[2] for probs in batch_probs]

		plt.plot(game_epochs, win_probs, label="Win Probability", color="g")
		# plt.plot(epochs, avg_reward, label="Average Reward", color="r")
		plt.ylim((0, 1.5))
		plt.xlabel('Epochs')
		plt.ylabel('Probability')
		# plt.ylabel('Average Reward')

		# pdb.set_trace()
		epochs = [i for i in range(num_epochs)]
		plt.title('DQN Agent Performance v/s Q-Learning Agent (epsilon-rate={0})'.format(epsilon_rate))
		plt.legend()
		plt.savefig('dqn_epsilon-rate={0}.png'.format(epsilon_rate))
		plt.close()

		plt.plot(epochs, avg_q, label="Average Q Value", color="b")
		plt.xlabel("Epochs")
		plt.ylabel("Q Value")
		plt.title('DQN Agent Performance v/s Q-Learning Agent (epsilon-rate={0})'.format(epsilon_rate))
		plt.legend()
		plt.savefig('dqn_avg_q.png')
		plt.close()

		plt.plot(epochs, avg_reward, label="Average Reward", color="b")
		plt.xlabel("Epochs")
		plt.ylabel("Reward")
		plt.title('DQN Agent Performance v/s Q-Learning Agent (epsilon-rate={0})'.format(epsilon_rate))
		plt.legend()
		plt.savefig('dqn_avg_reward.png')
		plt.close()

		model.save_weights('weights.dat')  # save model weights

		# final set of games to evaluate agent's performance after learning
		final_stats = self.measure_performance(game, 100)
		return np.array(win_probs).sum(), final_stats

	def play_game(self, agents):
		env = TicTacToeEnv()

		while True:
			agent = agent_by_mark(agents, str(env.show_turn()))
			ava_actions = env.available_actions()
			action = agent.act(ava_actions, env._world)

			# next_state = after_action_state(current_state, action)
			state, reward, done, info = env.step(action)
			if reward == 1:
				return 1 if agent == self else 2
			if done:
				return 0

	def act(self, ava_actions, state):
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

		# next_state = after_action_state(state, move)

		# TODO: check who, if anyone, has won
		# result = check_game_status(next_state)
		# if result != -1:
		# 	return 1 if result == self.mark else -1

		# TODO: Opp move

		# current_state = next_state

	def play_human(self):
		model = self.model
		model.load_weights("weights.dat")
		state = empty_state()  # reset game
		random_counter = 3  # number of random moves allowed in a game
		turn = 1

		print("\nBoard at turn {}".format(turn))
		# print_board(state)
		# TODO: Print board
		while True:

			human_move = int(input("Enter the cell you want to mark (0-15):"))
			row, col = int(human_move / len(state)), human_move % len(state)  # get the row and column to mark from the chosen action
			state[row, col] = 1
			turn += 1

			print("\nBoard at turn {}".format(turn))
			# print_board(state)
			# TODO: Print board

			# TODO: Check if this works
			winner = check_game_status(state)
			print("Winner is ", winner)
			if winner == 1:
				print("Congratulations! You have Won!")
				break
			elif winner == -1:
				print("Bummer! You lost to the AI!!")
				break

			q_values = model.predict(flatten_state(state))
			computer_move = int(np.argmax(q_values[0]))
			new_state = deepcopy(state)
			row, col = int(computer_move / len(state)), computer_move % len(state)  # get the row and column to mark from the chosen action
			new_state[row, col] = -1
			if np.array_equal(new_state, state):
				if random_counter > 0:
					empty_cells = get_available_actions(state)
					computer_move = choice(empty_cells)  # choose move randomly from available moves
					row, col = int(computer_move / len(state)), computer_move % len(state)  # get the row and column to mark from the chosen action
					new_state[row, col] = -1
					random_counter -= 1

				else:
					print("D'oh! The AI has much to learn still! Sorry for wasting your time, hooman!")
					break
			state = new_state
			turn += 1

			print("\nBoard at turn {}".format(turn))
			# print_board(state)
			# TODO: Print board

			# TODO: Verify this works
			winner = check_game_status(state)
			print("Winner is ", winner)
			if winner == 1:
				print("Congratulations! You have Won!")
				break
			elif winner == -1:
				print("Bummer! You lost to the AI!!")
				break

			if len(get_available_actions(state)) == 0:
				print("Alas! The Game has been drawn!")
				break

	def measure_performance(self, game, num_games):
		probs, games_played = [0, 0, 0], 0

		for i in range(num_games):
			# print("Starting Game {}".format(i+1))
			winner = self.play_game(game)

			if winner != -1:
				games_played += 1
				if winner == 0:
					probs[1] += 1.0
				elif winner == 1:
					probs[2] += 1.0
				else:
					probs[0] += 1.0
		# print("Ending Game {}".format(i+1))

		if games_played > 0:
			probs[0] = probs[0] * 1. / games_played
			probs[1] = probs[1] * 1. / games_played
			probs[2] = probs[2] * 1. / games_played

		return probs


def create_model():
	learning_rate = 0.001
	init = HeUniform()

	model = Sequential()
	model.add(Dense(24, input_shape=(27,), activation='relu', kernel_initializer=init))
	model.add(Dense(12, activation='relu', kernel_initializer=init))
	model.add(Dense(27, activation='linear', kernel_initializer=init))
	model.compile(loss=Huber, optimizer=Adam(learning_rate), metrics=['accuracy'])
	return model

def main():
	pass


def train():
	# hyperparameters
	gamma = 0.99  # discount factor for reward
	decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
	memory_size = 10000  # size of the replay memory
	batch_size = 1000

	input_size = 27
	model = create_model()
	# print(model.summary())
	# return
	dqn_agent = DQNAgent(model=model, memory_size=memory_size)

	# tdAgent2D = TDAgent(symbol=-1, is_learning=False, dims=2, build_states=False)
	# tdAgent2D.reset_agent()
	# TODO: Create RandomAgent or OpponentAgent
	cum_wins, final_stats = 0.0, []
	win_ratios = []
	game = TicTacToeEnv
	dqn_agent = DQNAgent(model=model, memory_size=memory_size)
	win_ratios.append(dqn_agent.train_network(game, batch_size=batch_size, num_epochs=10000, gamma=gamma))
	# win_ratios.append(dqn_agent.train_network(tictactoe, batch_size=batch_size, num_epochs=10000, gamma=gamma, epsilon_rate=1))
	# win_ratios.append(dqn_agent.train_network(tictactoe, batch_size=batch_size, num_epochs=10000, gamma=gamma, epsilon_rate=0.8))
	cum_wins, final_stats = dqn_agent.train_network(tdAgent2D, batch_size=batch_size, num_epochs=1000, gamma=gamma, epsilon_rate=0.3)
	print("\nTotal cumulative wins with heuristic: ", cum_wins)
	print("\n Final Matchup Results with heuristic: {0} Wins | {1} Draws | {2} Losses".format(final_stats[2], final_stats[1], final_stats[0]))
	print("\nArchitecture Details:")
	print("Batch size {0} |  Memory size: {1}".format(batch_size, memory_size))
	dqn_agent.play_human(2)


if __name__ == "__main__":
	train()
