from stable_baselines3 import SAC

# 加载预训练模型
model = SAC.load("pretrained_model.zip")

# 设置新的环境
model.set_env(new_env)

# 继续训练
model.learn(total_timesteps=1e5)

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

# 创建replay buffer并加载专家数据
buffer = ReplayBuffer(buffer_size=1e5, ...)
# 加载专家数据到buffer...

# 使用专家数据预训练
model = SAC("MlpPolicy", env)
model.replay_buffer = buffer
model.learn(total_timesteps=1e4)

# 继续标准RL训练
model.learn(total_timesteps=1e5)