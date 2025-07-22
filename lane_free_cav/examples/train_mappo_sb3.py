import supersuit as ss
from stable_baselines3 import PPO
from pettingzoo.utils import parallel_to_aec
from lane_free_cav.envs.lane_free_cav_v0 import LaneFreeCAVEnv

base_env  = LaneFreeCAVEnv(render_mode=None)
base_env  = add_spacing_info(base_env)
vec_env   = ss.concat_vec_envs_v1(
               ss.pettingzoo_env_to_vec_env_v1(base_env),
               num_vec_envs=4, num_cpus=1, base_class="stable_baselines3")

model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=200_000)
model.save("ppo_lane_free")
