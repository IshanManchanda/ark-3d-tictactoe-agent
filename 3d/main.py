from random import shuffle

import numpy as np
from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv, agent_by_mark

from agent import DQNAgent


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


def main():
	# Driver code to run 3D human-vs-AI TicTacToe
	# Create environment
	env = TicTacToeEnv()

	# Assign player 1 and 2 randomly to human and agent
	marks = ['1', '2']
	shuffle(marks)
	agents = [HumanAgent(marks[0]), DQNAgent(marks[1])]
	print(f'Human: Player {marks[0]}. Machine: Player {marks[1]}')

	# Counter for moves to check if game ended in draw
	moves = 0

	while True:
		# Get the player to move
		agent = agent_by_mark(agents, str(env.show_turn()))

		# Get possible moves for this player and ask for chosen move
		ava_actions = env.available_actions()
		action = agent.act(ava_actions, np.array(env._world))

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
