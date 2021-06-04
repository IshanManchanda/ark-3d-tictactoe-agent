import sys
from random import shuffle

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
			inp = input('Enter position [00 - 22], q for quit: ')
			if inp.lower() == 'q':
				return None

			try:
				# Test if input if valid
				action = '0' + inp
				if action not in ava_actions:
					raise ValueError()

			# Continue iterating if input invalid, otherwise break and return
			except ValueError:
				print(f"Illegal position: '{inp}'")
			else:
				break

		return self.mark + action


class AIAgent:
	# Agent that implements the minimax algorithm

	# Array that stores evaluations for positions
	dp = [None] * 3 ** 9

	def __init__(self, mark):
		self.mark = mark

	def act(self, ava_actions, state):
		# Variables to hold the current best action and its evaluation
		best_action, best_eval = None, float('-inf')

		# Opponent's mark
		opp_mark = '2' if self.mark == '1' else '1'
		alpha, beta = float('-inf'), float('inf')

		# Iterate over all possible moves
		for action in AIAgent.get_available_actions(state):
			# Get the state after performing the move
			child = after_action_state(state, self.mark + action)

			# Use the negamax variant of the minimax algorithm to get
			# the evaluation of the state after the selected move
			child_eval = -AIAgent.negamax(child, 8, alpha, beta, opp_mark)

			# If this is better than our current best move, select it
			if child_eval > best_eval:
				best_eval = child_eval
				best_action = action

		# Return the selected best action
		return self.mark + best_action

	@staticmethod
	def negamax(state, depth, alpha, beta, mark):
		# We use the negamax variant of the minimax algorithm as it's more
		# convenient and applies perfectly well since TTT is a zero-sum game.
		# The algorithms are equivalent, with the only difference being
		# implementation wherein we simply take maximums
		# (instead of alternating minimums and maximums)
		# by multiplying the evaluation by -1 at each level.

		# Check if we have already computed this state.
		# We map each state to an index and use that for the dp table.
		state_idx = AIAgent.get_idx(state)
		if AIAgent.dp[state_idx] is not None:
			# If we have the evaluation, simply return it
			return AIAgent.dp[state_idx]

		# Check if the game is won/lost in the current state
		status = check_game_status(state)
		if status != -1:
			# If we are the winning player, return a positive evaluation
			if status == mark:
				# The evaluation decreases with the number of levels needed
				# to achieve it, incentivising the algorithm to win quickly
				AIAgent.dp[state_idx] = 1 + depth
				return AIAgent.dp[state_idx]

			# Similarly, we incentivize loser later rather than earlier
			AIAgent.dp[state_idx] = -1 - depth
			return AIAgent.dp[state_idx]

		# If this is not a winning/losing position and the depth is 0,
		# the game must be a draw. Return a 0 evaluation
		if depth == 0:
			AIAgent.dp[state_idx] = 0
			return 0

		# Current maximum evaluation and opponent's mark
		max_eval = float('-infinity')
		opp_mark = '2' if mark == '1' else '1'

		# Iterate over all possible moves
		for action in AIAgent.get_available_actions(state):
			# Get the evaluation of the state after the current action
			child = after_action_state(state, mark + action)
			pos_eval = -AIAgent.negamax(child, depth - 1, opp_mark)

			# Update our maximum if needed
			max_eval = max(max_eval, pos_eval)

		# Store the computed evaluation and return it
		AIAgent.dp[state_idx] = max_eval
		return max_eval

	@staticmethod
	def get_idx(state):
		# Helper function to map states to unique ids in the DP table

		# Realize that the board is entirely represented by 9 ordered variables
		# that take on values 0, 1, and 2
		# This allows us to conveniently generate one-one indexes using
		# ternary numbers (base 3)
		idx = 0
		for i in range(3):
			for j in range(3):
				idx += state[0][i][j] * 3 ** (3 * i + j)
		return idx

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
				# Check if position vacant and add it as a valid move
				if state[0][i][j] == 0:
					available.append(f'{0}{i}{j}')
		return available


def main():
	# Driver code to run the 2D human-vs-AI TicTacToe
	env = TicTacToeEnv()
	marks = ['1', '2']
	shuffle(marks)
	agents = [HumanAgent(marks[0]), AIAgent(marks[0])]
	episode = 0
	done = False
	moves = 0
	while not done:
		agent = agent_by_mark(agents, str(env.show_turn()))
		print(agent.mark)
		ava_actions = env.available_actions()
		action = agent.act(ava_actions, env._world)
		print(action)
		if action is None:
			sys.exit()

		state, reward, done, info = env.step(action)

		print()
		env.render()
		if done:
			env.show_result()
			break
		moves += 1
		if moves == 9:
			print("Draw.")
			break
	episode += 1


if __name__ == '__main__':
	main()
