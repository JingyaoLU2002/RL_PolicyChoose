import gym
from gym.envs.registration import register

# 注册环境
register(
    id = 'lawyer_suspector_env_v0',
    entry_point='Lawyer-Suspector-Env.lawyer-suspector-env:LawyerSuspectEnv',
)

# 查询测试
env = gym.spec('lawyer_suspector_env_v0')
print(env)