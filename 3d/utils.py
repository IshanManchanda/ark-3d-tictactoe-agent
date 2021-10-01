import numpy as np


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
