import numpy as np
import gymnasium as gym
import tqdm
import cv2

seed = 11
np.random.seed(seed)

class CartPoleAgent:

	def __init__(
		self,
		epsilon=0.15,
		gamma=0.9,
		lr=0.05,
		discretization_factor=100000,
		max_steps=1000,
		max_episodes=1000000,
		render_episode=-1
	):
		self.env = gym.make("CartPole-v1", render_mode="rgb_array")
		self.d_factor = discretization_factor
		self.Q, self.unit = self._discretize()
		self.epsilon = epsilon
		self.gamma = gamma
		self.lr = lr
		self.max_steps = max_steps
		self.max_episodes = max_episodes
		self.render_episode = render_episode

	def _discretize(self):
		
		unit = self.env.observation_space.high/self.d_factor
		# manually set velocity and angular velocity unit (max position ~ 10*[max angle] -> max velocity ~ 10*[max velocity])
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

		if self.render_episode >= 0 and episode % self.render_episode == 0:
			print(len(self.Q))
			img = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
			cv2.imshow("CartPole-v1", img)
			cv2.waitKey(50)

	def choose_action(self, state, on_policy=True):

		if state not in self.Q:
			self.Q[state] = np.random.rand(2)

		else:
			print("Já vi.")

		q = self.Q[state]

		if on_policy:
			action = self.env.action_space.sample() \
						if np.random.uniform(0, 1) < self.epsilon \
						else np.argmax(q)

		else:
			action = np.argmax(q)

		return action

	def sarsa(self):

		for episode in tqdm.tqdm(range(self.max_episodes)):
			state, _ = self.env.reset()
			state = self.discretize(state)
			action = self.choose_action(state, on_policy=True)

			for _ in range(self.max_steps):
				_state, reward, terminated, truncated, info = self.env.step(action)

				if terminated or truncated:
					break

				self.render(episode)

				_state = self.discretize(_state)
				_action = self.choose_action(_state, on_policy=True)

				q = self.Q[state][action]
				self.Q[state][action] = q + self.lr*(reward + self.gamma*self.Q[_state][_action] - q)
				state = _state
				action = _action

	def q_learning(self):

		for episode in tqdm.tqdm(range(self.max_episodes)):
			state, _ = self.env.reset()
			state = self.discretize(state)

			for _ in range(self.max_steps):
				action = self.choose_action(state, on_policy=True)
				_state, reward, terminated, truncated, info = self.env.step(action)

				if terminated or truncated:
					break

				self.render(episode)

				_state = self.discretize(_state)

				q = self.Q[state][action]
				max_q_action = self.choose_action(action, on_policy=False)
				self.Q[state][action] = q + self.lr*(reward + self.gamma*max_q_action - q)
				state = _state

def main():
	agent = CartPoleAgent(render_episode=10000)
	agent.sarsa()

if __name__ == "__main__":
	main()