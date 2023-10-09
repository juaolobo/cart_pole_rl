import numpy as np
import gymnasium as gym
import tqdm

seed = 11
np.random.seed(seed)

class CartPoleAgent:

	def __init__(self, discretization_factor=100000, max_steps=100, max_episodes=10000):
		self.env = gym.make("CartPole-v1")
		self.n_states = discretization_factor
		self.Q, self.unit = self._discretize()
		self.epsilon = 0.15
		self.gamma = 0.9
		self.lr = 0.01
		self.max_steps = max_steps
		self.max_episodes = max_episodes

	def _discretize(self):
		
		unit = self.env.observation_space.high/self.n_states
		Q = {}

		return Q, unit

	def discretize(self, state):

		discrete_state = np.floor(state/(2*self.unit))

		return tuple(discrete_state.astype(int))

	def choose_action(self, state):

		if state not in self.Q:
			self.Q[state] = np.random.rand(2)

		q = self.Q[state]

		action = self.env.action_space.sample() \
					if np.random.uniform(0, 1) < self.epsilon \
					else np.argmax(q)

		return action

	def sarsa(self):

		for episode in tqdm.tqdm(range(self.max_episodes)):
			state, _ = self.env.reset()
			state = self.discretize(state)
			action = self.choose_action(state)

			for _ in range(self.max_steps):
				_state, reward, terminated, truncated, info = self.env.step(action)
				_state = self.discretize(_state)
				_action = self.choose_action(_state)

				q = self.Q[state][action]
				self.Q[state][action] = q + self.lr*(reward + self.gamma*self.Q[_state][_action] - q)
				state = _state
				action = _action

	def q_learning(self):

		for episode in range(self.max_episodes):		
			state, _ = self.env.reset()

			for _ in range(self.max_steps):
				action = self.choose_action(state)
				_state, reward, terminated, truncated, info = env.step(action)
				q = self.Q[state][action]
				max_q_action = np.argmax(self.Q[_state])
				self.Q[state][action] = q + self.lr*(reward + self.gamma*max_q_action - q)
				state = _state

def main():
	agent = CartPoleAgent()
	agent.sarsa()
	breakpoint()

if __name__ == "__main__":
	main()