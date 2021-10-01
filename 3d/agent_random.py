import numpy as np


class RandomAgent:
	# RandomAgent class, used to train DQN Agent
	def __init__(self, mark):
		self.mark = mark

	def set_mark(self, mark):
		# Function to update mark for training
		self.mark = mark

	def act(self, ava_actions, state):
		# Sample one action from all available actions and return action string
		idx = np.random.choice(len(ava_actions))
		return self.mark + ava_actions[idx]
