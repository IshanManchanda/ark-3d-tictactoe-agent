from random import shuffle

import numpy as np
from gym_tictactoe.envs.tictactoe_env import TicTacToeEnv, agent_by_mark

from agent_dqn import DQNAgent
from agent_human import HumanAgent


def main():
	# TODO: Load trained DQN agent from disk or exit if none
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
