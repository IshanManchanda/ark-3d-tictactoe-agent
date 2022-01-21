# ARK Task 5: 3D Tic Tac Toe Agent

__Note__: Please refer the
[documentation file](https://github.com/IshanManchanda/ark-submission/blob/master/ARK_Documentation.pdf)
for a thorough discussion of the problem statement,
approaches, and implementation details.

### Project Structure

The project has 2 independent components that solve the 2d and 3d cases.

The 2d case has a single ```2d.py``` file which contains both the agent
and the driver program that allows agent-vs-human play.

The solution for the 3D variant is incomplete and located in the 3d folder.
The entry-point into this part is the ```main.py``` file which allows (will allow)
agent-vs-human play. It imports the ```DQNAgent``` from the ```agent.py``` file
which has the code for the agent and is used for training.
The ```agent.py``` file imports the ```Memory``` class from the ```memory.py``` file which is
used for the Experience Replay feature.

As mentioned in the documentation, a large part of the required components are
implemented, except for the training pipeline. The documentation thus contains
an in-depth explanation of the approach and also delves into implementation
details.
