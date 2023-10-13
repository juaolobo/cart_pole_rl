import gymnasium as gym

import numpy as np
import cv2

import tqdm
import matplotlib.pyplot as plt

seed = 11
np.random.seed(seed)

class CartPoleAgent:

	def __init__(
		self,
		epsilon=0.25,
		gamma=0.9,
		lr=0.1,
		discretization_factor=100,
		render_episode=-1
	):
		self.env = gym.make("CartPole-v1", render_mode="rgb_array")
		self.d_factor = discretization_factor
		self.Q, self.unit = self._discretize()
		self.epsilon = epsilon
		self.gamma = gamma
		self.lr = lr
		self.render_episode = render_episode
		self.reward_history = {"q_learning": [], "sarsa": []}

	def _discretize(self):
		
		unit = self.env.observation_space.high/self.d_factor
		# manually set velocity and angular velocity unit (max position ~ 10*[max angle] -> max velocity ~ 10*[max ang velocity])
		velocity_unit = 1000/self.d_factor
		ang_velocity_unit = velocity_unit/10

		unit[1] = velocity_unit
		unit[3] = ang_velocity_unit
		Q = {}

		return Q, unit

	def discretize(self, state):

		# np.fix == np.floor towards zero
		discrete_state = np.fix(state/(self.unit))

		return tuple(discrete_state.astype(int))

	def render(self, episode):

		if self.render_episode > 0 and episode % self.render_episode == 0:

			img = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
			cv2.imshow("CartPole-v1", img)
			cv2.waitKey(50)

	def choose_action(self, state, on_policy=True):

		if state not in self.Q:
			self.Q[state] = [0,0]
			#self.Q[state] = (np.random.rand(2) * 2) - 1

		q = self.Q[state]

		if on_policy:
			action = self.env.action_space.sample() \
						if np.random.uniform(0, 1) < self.epsilon \
						else np.argmax(q)

		else:
			action = np.argmax(q)

		return action

	def sarsa(self, max_episodes, max_steps):

		for episode in tqdm.tqdm(range(max_episodes)):
			state, _ = self.env.reset()
			state = self.discretize(state)
			action = self.choose_action(state, on_policy=True)

			reward_per_ep = 0
			for _ in range(max_steps):
				_state, reward, terminated, truncated, info = self.env.step(action)
				reward_per_ep += reward

				if terminated or truncated:
					break

				self.render(episode)

				_state = self.discretize(_state)
				_action = self.choose_action(_state, on_policy=True)

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
				action = self.choose_action(state, on_policy=True)
				_state, reward, terminated, truncated, info = self.env.step(action)
				reward_per_ep += reward

				if terminated or truncated:
					break

				self.render(episode)

				_state = self.discretize(_state)
				max_action = self.choose_action(_state, on_policy=False)
				max_q_action = self.Q[_state][max_action]

				q = self.Q[state][action]
				self.Q[state][action] = q + self.lr*(reward + self.gamma*max_q_action - q)
				state = _state

			self.reward_history["q_learning"].append(reward_per_ep)

	def simulate_episode(self, steps):

		self.render_episode = 1
		state, _ = self.env.reset()

		while True:
			state = self.discretize(state)
			action = self.choose_action(state, on_policy=True)
			_state, reward, terminated, truncated, info = self.env.step(action)

			if terminated or truncated:
				return

			self.render(0)
			state = _state

	def save_plot(self, filename):

		labels = {"q_learning": "Q Learning", "sarsa": "SARSA"}
		colors = {"q_learning": "blue", "sarsa": "red"}

		for algo, rewards in self.reward_history.items():
			plt.plot(range(len(rewards)), rewards, label=labels[algo], color=colors[algo])

		plt.legend()
		# plt.savefig(filename)
		plt.show()


def main():
	agent = CartPoleAgent(epsilon=0.25)
	# agent.sarsa(max_episodes=50000, max_steps=500)
	agent.q_learning(max_episodes=10000, max_steps=500)
	agent.save_plot('teste.png')
	breakpoint()
	# agent.simulate_episode(10000)

if __name__ == "__main__":
	main()