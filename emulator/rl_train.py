# rl_train.py
import pdb
from rl_futures_env import FuturesRLEnv
from kichaos.stable3 import PPO # 假设这是你的 SB3 库

if __name__ == "__main__":
    # pdb.set_trace() # 注释掉 pdb，除非需要调试
    env = FuturesRLEnv(code='IM',
                       start_date='2025-03-03',
                       end_date='2025-03-10',
                       uri="temp")

    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=10000)

    # 评估模型
    print("\n--- Evaluating Trained Model ---")
    obs = env.reset()
    for _ in range(500): # 评估 500 步
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()