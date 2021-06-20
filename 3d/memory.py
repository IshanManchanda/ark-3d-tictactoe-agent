import numpy as np


class Memory:
	# Helper class for the memory required for Experience Replay
	def __init__(self, memory_size=100):
		self.memory = [None, None, None, None]
		self.max_size = memory_size

		self.input_shape = None
		self.size = 0

	def reset_memory(self):
		self.memory = [None, None, None, None]

	def remember(self, current_state, action, next_state, reward):
		# Check if memory arrays have not been initialized
		if self.size == 0:
			# Create arrays with one extra dimension
			self.memory[0] = np.zeros((1, *current_state.shape))
			self.memory[1] = np.zeros((1, *action.shape))
			self.memory[2] = np.zeros((1, *next_state.shape))
			self.memory[3] = np.zeros((1, *reward.shape))

			# Copy over the data to the first element
			self.memory[0][0] = current_state.copy()
			self.memory[1][0] = action.copy()
			self.memory[2][0] = next_state.copy()
			self.memory[3][0] = reward.copy()
			self.size += 1
			return

		# If the memory is at max capacity, we remove the oldest value
		if self.size == self.max_size:
			self.memory[0] = np.stack((*self.memory[0][1:], current_state))
			self.memory[1] = np.stack((*self.memory[1][1:], action))
			self.memory[2] = np.stack((*self.memory[2][1:], next_state))
			self.memory[3] = np.stack((*self.memory[3][1:], reward))
			return

		# Otherwise simply append the new data along the additional dimension
		# Unfortunately numpy is not able to automatically promote the data
		# for appending.
		self.memory[0] = np.stack((*self.memory[0], current_state))
		self.memory[1] = np.stack((*self.memory[1], action))
		self.memory[2] = np.stack((*self.memory[2], next_state))
		self.memory[3] = np.stack((*self.memory[3], reward))
		self.size += 1

	def sample(self, batch_size):
		# Check if the current memory size is less than the batch size
		if self.size == 0:
			return None
		if self.size <= batch_size:
			return [x.copy() for x in self.memory]

		# Sample indices and pick corresponding elements
		indices = np.random.choice(self.size, batch_size)
		return [x[indices].copy() for x in self.memory]
