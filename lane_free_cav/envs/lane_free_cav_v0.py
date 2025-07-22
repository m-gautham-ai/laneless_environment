from pettingzoo.utils.env import ParallelEnv
import gymnasium as gym
from gymnasium import spaces
from .dynamics import VehicleModel
from lane_free_cav.utils.renderer import Viewer
from lane_free_cav.wrappers.spacing_info import add_spacing_info
import numpy as np

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
    ):
        super().__init__()
        self.dt = dt
        self.road_width = road_width
        self.max_cycles = max_cycles
        self.render_mode = render_mode
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
        obs_dim = 8  # [x,y,vx,vy, heading, size_x, size_y, min_dist]
        self._obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # ax, ay
        self.observation_spaces = {a: self._obs_space for a in self.agents}
        self.action_spaces = {a: self._act_space for a in self.agents}
        # GUI
        self.viewer = Viewer(road_width, render_mode) if render_mode else None
        # bookkeeping
        self.step_count = 0

    def reset(self, seed=None, options=None):

        self.step_count = 0
        self.agents = self.possible_agents[:]
        # random spawn
        for m in self._models.values():
            m.reset(x=np.random.uniform(0, self.road_width),
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
        # spacing calc & reward
        for a in self.agents:
            min_d = self._min_distance(a)
            infos[a] = {"nearest_dist": min_d}
            obs[a]   = self._get_obs(a, min_d)
            rew[a]   = -1.0 if min_d < 1.0 else 0.1   # simple reward
            done[a]  = term
            trunc[a] = False
        # safety override (optional)
        # self._apply_cbf(actions)
        if self.render_mode == "human":
            self.render()
        if term:
            self.agents = []
        return obs, rew, done, trunc, infos

    def _get_obs(self, agent, dist=None):
        m = self._models[agent]
        size = self.catalog[self.types[agent]]
        return np.array([m.x, m.y, m.vx, m.vy, m.heading,
                         size["L"], size["W"],
                         dist if dist is not None else self._min_distance(agent)],
                         dtype=np.float32)

    def _min_distance(self, agent):
        m = self._models[agent]
        dists = [m.distance_to(self._models[o]) for o in self.agents if o != agent]
        return min(dists) if dists else np.inf

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.viewer is None:
            self.viewer = Viewer(self.road_width, self.render_mode)

        self.viewer.draw(self._models, self.types)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space
