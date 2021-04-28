# dqn_atari_her
In this project the influence of Hindsight Experience Replay (HER) on a DQN can be tested on the Atari games Montezuma's Revenge and Ms. Pac-Man.
It uses a modified version of the Baselines package from openAI found in the .rar file.

The file dqn.py executes the normal DQN algorithm which can for example be tested on the game Breakout. Here it yields nice results.

The file dqn_her.py adds the Hindsight Experience Replay (HER) extension. Unfortunately it was observed that this doesn't improve the mean reward on Montezuma's Revenge which stays zero and even reduces the mean reward on the game Ms. Pac-Man compared to the vanilla version in the dqn.py file.

This might be due to the problem that HER is just not supposed to work on the Atari games on the DQNs or maybe due to an error in the code.
I'd guess its due to the first problem because the game Montezuma's Revenge was only solved in February 2021 in the Paper "First return, then explore" found here: https://www.nature.com/articles/s41586-020-03157-9

Contributions are nevertheless welcome and maybe we can solve the Atari games with HER nevertheless! :D
