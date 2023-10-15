import gymnasium as gym

import numpy as np
import cv2

import tqdm
import matplotlib.pyplot as plt
import argparse

import sys

seed = 11
np.random.seed(seed)

class CartPoleAgent:

	def __init__(
		self,
		epsilon=0.25,
		gamma=0.9,
		learning_rate=0.2,
		render_episode=-1
	):
		self.env = gym.make("CartPole-v1", render_mode="rgb_array")
		self.obs_discretization = (6, 6, 12, 12)
		self.Q, self.unit = self._discretize()
		self.epsilon = epsilon
		self.gamma = gamma
		self.lr = learning_rate
		self.render_episode = render_episode
		self.reward_history = {"q_learning": [], "sarsa": []}

	def _discretize(self):

		state_dim = np.array(self.obs_discretization) + 1
		q_space_dims = (*state_dim, ) + (self.env.action_space.n,)

		Q = np.zeros(q_space_dims)
		unit = np.array([2.4, 3, 0.2095, 1])

		unit = 2*unit/self.obs_discretization

		return Q, unit

	def fix_state(self, state):

		state[1] = 3 if state[1] > 3 else state[1]
		state[1] = -3 if state[1] < -3 else state[1]

		state[3] = 1 if state[3] > 1 else state[3]
		state[3] = -1 if state[3] < -1 else state[3]

		return state		

	def discretize(self, state):

		state = self.fix_state(state)

		# np.fix == np.floor towards zero
		discrete_state = np.floor(state/(self.unit)) + np.array(self.obs_discretization)/2

		return tuple(discrete_state.astype(int))

	def render(self, episode):

		if self.render_episode > 0 and episode % self.render_episode == 0:

			img = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
			cv2.imshow("CartPole-v1", img)
			cv2.waitKey(20)

	def choose_action(self, state):

		q = self.Q[state]

		action = self.env.action_space.sample() \
					if np.random.uniform(0, 1) < self.epsilon \
					else np.argmax(q)

		return action

	def sarsa(self, max_episodes, max_steps):

		for episode in tqdm.tqdm(range(max_episodes)):

			state, _ = self.env.reset()
			state = self.discretize(state)

			reward_per_ep = 0
			for _ in range(max_steps):
				action = self.choose_action(state)
				_state, reward, terminated, truncated, info = self.env.step(action)
				reward_per_ep += reward

				if terminated or truncated:
					break

				self.render(episode)

				_state = self.discretize(_state)
				_action = self.choose_action(_state)

				q = self.Q[state][action]
				self.Q[state][action] = q + self.lr*(reward + self.gamma*self.Q[_state][_action] - q)
				state = _state
				action = _action

			self.reward_history["sarsa"].append(reward_per_ep)

	def q_learning(self, max_episodes, max_steps):

		for episode in tqdm.tqdm(range(max_episodes)):
			state, _ = self.env.reset()
			state = self.discretize(state)

			reward_per_ep = 0
			for _ in range(max_steps):

				action = self.choose_action(state)
				_state, reward, terminated, truncated, info = self.env.step(action)

				reward_per_ep += reward

				if terminated or truncated:
					break

				self.render(episode)

				_state = self.discretize(_state)
				max_q_action = np.max(self.Q[_state])

				q = self.Q[state][action]
				self.Q[state][action] = q + self.lr*(reward + self.gamma*max_q_action - q)
								
				state = _state

			if episode % 100 == 0:
				self.reward_history["q_learning"].append(self.trial_run())

	def trial_run(self):

		total_reward = 0
		state, _ = self.env.reset()
		while True:
			state = self.discretize(state)
			action = np.argmax(self.Q[state])
			_state, reward, terminated, truncated, info = self.env.step(action)

			total_reward += reward

			if terminated or truncated:
				break

			state = _state

		return total_reward

	def simulate_episode(self, steps):

		self.render_episode = 1
		state, _ = self.env.reset()

		while True:
			state = self.discretize(state)
			action = np.argmax(self.Q[state])
			_state, reward, terminated, truncated, info = self.env.step(action)

			if terminated or truncated:
				return

			self.render(0)
			state = _state

	def save_plot(self, filename):

		labels = {"q_learning": "Q Learning", "sarsa": "SARSA"}
		colors = {"q_learning": "blue", "sarsa": "red"}

		for algo, rewards in self.reward_history.items():
			fig = plt.figure()
			plt.plot(range(len(rewards)), rewards, label=labels[algo], color=colors[algo])
			plt.legend()
			fig.savefig(filename.replace(".png", f"_{algo}.png"))
			plt.close()

def parse_params():

	parser = argparse.ArgumentParser(description='Agent parameters')

	parser.add_argument('-lr', '--learning_rate', type=float, nargs=1,
                    help='Taxa de aprendizado')

	parser.add_argument('-e', '--epsilon', type=float, nargs=1,
                    help='Taxa de exploração')

	parser.add_argument('-g', '--gamma', type=float, nargs=1,
                    help='Desconto da recompensa')

	return parser.parse_args()


def main():

	if len(sys.argv) > 1:
		kwargs = dict(map(lambda x : (x[0], *x[1]) if x[1] != None else x, parse_params()._get_kwargs()))
		agent = CartPoleAgent(**kwargs)
		output_file = f'plots/lr_{kwargs["learning_rate"]}_eps_{kwargs["epsilon"]}_gamma_{kwargs["gamma"]}.png'

	else:
		agent = CartPoleAgent()
		output_file = f'plots/default.png'

	# agent.sarsa(max_episodes=50000, max_steps=500)
	agent.q_learning(max_episodes=300000, max_steps=500)

	agent.simulate_episode(1000)
	agent.save_plot(output_file)

if __name__ == "__main__":
	main()