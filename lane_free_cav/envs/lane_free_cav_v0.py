from pettingzoo.utils.env import ParallelEnv
import gymnasium as gym
from gymnasium import spaces
from .dynamics import VehicleModel
from lane_free_cav.utils.renderer import Viewer
from lane_free_cav.wrappers.spacing_info import add_spacing_info
import numpy as np
import warnings


# x: The agent's longitudinal position on the road.
# y: The agent's lateral position on the road.
# vx: The agent's longitudinal velocity.
# vy: The agent's lateral velocity.
# heading: The agent's heading angle.
# length: The length of the agent's vehicle.
# width: The width of the agent's vehicle.
# min_d : The minimum distance to the nearest other vehicle.

class LaneFreeCAVEnv(ParallelEnv):
    metadata = {
        "name": "lane_free_cav_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        n_agents=10,
        road_width=60.0,
        dt=0.1,
        vehicle_types=None,
        max_cycles=1000,
        render_mode=None,
        screen_scale=10, # pixels per meter
    ):
        super().__init__()
        self.dt = dt
        self.road_width = road_width
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.screen_scale = screen_scale
        self.viewer = None
        # vehicle catalogue
        self.catalog = {
            "car": {"L": 4.7, "W": 1.8, "max_acc": 3.5},
            "truck": {"L": 8.0, "W": 2.5, "max_acc": 2.0},
            "moto": {"L": 2.2, "W": 0.6, "max_acc": 4.0},
        }
        self.possible_agents = [f"veh_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]
        # pick types
        self.types = {a: np.random.choice(vehicle_types or list(self.catalog)) for a in self.agents}
        # dynamics objects
        self._models = {a: VehicleModel(self.catalog[self.types[a]], dt) for a in self.agents}
        # spaces
        # Define observation and action spaces for a single agent
        # obs: [x, y, vx, vy, heading, L, W, min_d]
        obs_low = np.array([-1000, 0, -100, -100, -np.pi, 0, 0, 0], dtype=np.float32)
        obs_high = np.array([1000, self.road_width, 100, 100, np.pi, 10, 5, 1000], dtype=np.float32)
        self._single_observation_space = spaces.Box(obs_low, obs_high)
        self._single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Create spaces dictionary for all possible agents
        self.observation_spaces = spaces.Dict(
            {a: self._single_observation_space for a in self.possible_agents}
        )
        self.action_spaces = spaces.Dict(
            {a: self._single_action_space for a in self.possible_agents}
        )
        # GUI
        self.viewer = Viewer(road_width, render_mode) if render_mode else None
        # bookkeeping
        self.step_count = 0

    def reset(self, seed=None, options=None):

        self.step_count = 0
        self.agents = self.possible_agents[:]
        # random spawn
        for m in self._models.values():
            m.reset(x=0.0,  # Always start from the left side
                    y=np.random.uniform(0, self.road_width),
                    vx=20.0, vy=0, heading=0)
        obs = {a: self._get_obs(a) for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, info

    def step(self, actions):
        assert actions.keys() == set(self.agents)
        self.step_count += 1
        term = self.step_count >= self.max_cycles
        obs, rew, done, trunc, infos = {}, {}, {}, {}, {}
        # physics update
        for a, act in actions.items():
            self._models[a].step(act)             # apply accel

        agents_to_remove = set()
        # Check for collisions between agents
        agent_list = list(self.agents)
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a1 = agent_list[i]
                a2 = agent_list[j]
                dist = np.linalg.norm([self._models[a1].x - self._models[a2].x, self._models[a1].y - self._models[a2].y])
                if dist < 5.0:  # Collision threshold
                    agents_to_remove.add(a1)
                    agents_to_remove.add(a2)

        # Rewards
        # Reward for forward velocity, with a penalty for each step taken
        step_penalty = 0.1
        rewards = {a: (self._models[a].vx * self.dt) - step_penalty for a in self.agents}

        # Boundary checks and wrap-around
        agents_to_remove = set()
        for a in self.agents:
            if a in agents_to_remove:
                rewards[a] = -10.0 # Penalty for crashing
                done[a] = True
                trunc[a] = False
                obs[a] = self._get_obs(a, 0.0)
                infos[a] = {'status': 'collided'}
                continue

            m = self._models[a]
            # Check for successful completion by reaching the right side
            if m.x > self.road_width:
                agents_to_remove.add(a)
                rewards[a] = 100.0  # Large reward for completion
                done[a] = True
                trunc[a] = False
                obs[a] = self._get_obs(a, self._min_distance(a))
                infos[a] = {'status': 'completed'}
                continue

            # Terminate if touching lateral boundaries (sand)
            if m.y - m.width / 2 < 0 or m.y + m.width / 2 > self.road_width:
                agents_to_remove.add(a)
                rewards[a] = -10.0 # Penalty for going off-road
                done[a] = True
                trunc[a] = False
                obs[a] = self._get_obs(a, 0.0)
                infos[a] = {'status': 'crashed'}
                continue

            # Periodic boundary (longitudinal)
            old_x = m.x
            m.x %= self.road_width
            if m.x < old_x:  # Agent has wrapped around
                m.y = np.random.uniform(0, self.road_width)

            # Spacing calc & reward for agents still alive
            min_d = self._min_distance(a)
            infos[a] = {"nearest_dist": min_d}
            obs[a] = self._get_obs(a, min_d)
            done[a] = term
            trunc[a] = term

        # Remove agents that have crashed or collided
        for a in agents_to_remove:
            if a in self.agents:
                self.agents.remove(a)
        # safety override (optional)
        # self._apply_cbf(actions)
        if self.render_mode == "human":
            self.render()
        if term:
            self.agents = []
        return obs, rewards, done, trunc, infos

    def _get_obs(self, agent, min_d=0.0):
        m = self._models[agent]
        return np.array([m.x, m.y, m.vx, m.vy, m.heading,
                         m.length, m.width, min_d], dtype=np.float32)

    def _min_distance(self, agent):
        m = self._models[agent]
        dists = [np.linalg.norm([m.x-o.x, m.y-o.y]) for o in self._models.values() if o is not m]
        return min(dists) if dists else 1000.0

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.viewer is None:
            self.viewer = Viewer(self.road_width, self.render_mode, scale=self.screen_scale)

        active_models = {a: self._models[a] for a in self.agents}
        self.viewer.draw(active_models, self.types)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
