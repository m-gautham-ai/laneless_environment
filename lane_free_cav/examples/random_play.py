import lane_free_cav as lfc
from lane_free_cav.envs.lane_free_cav_v0 import LaneFreeCAVEnv

env = LaneFreeCAVEnv(render_mode="human", n_agents=20, road_width=30.0)
obs, info = env.reset(seed=42)
for _ in range(1000):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
    if not env.agents:
        obs, info = env.reset()
env.close()
