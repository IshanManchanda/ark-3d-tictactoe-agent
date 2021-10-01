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
