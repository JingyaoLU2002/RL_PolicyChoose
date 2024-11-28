import os
import numpy as np
import gym
from gym.envs.registration import register
from tqdm import tqdm
from DB import DB
from agent import Agent
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# PPO策略网络类
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim,action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc4 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)  # 输出动作的概率分布

# PPO价值网络类
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc4 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# PPO算法主体
class PPO:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, checkpoint_dir):
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.checkpoint_dir = checkpoint_dir

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_model(self, actor_path, critic_path, return_list, episode,case_index):
        # 保存模型及进度
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        np.savez(self.checkpoint_dir + f"/checkpoint.npz", return_list=return_list, episode=episode,case_index = case_index)

    def save_best_model(self, actor_path, critic_path):
        # 保存最优模型
        torch.save(self.actor.state_dict(), actor_path)  # 保存最优actor模型
        torch.save(self.critic.state_dict(), critic_path)  # 保存最优critic模型
        np.savez(self.checkpoint_dir + f"/checkpoint_best.npz")

    def load_model(self, actor_path, critic_path, checkpoint_path):
        self.actor.load_state_dict(torch.load(actor_path))  # 加载actor模型
        self.critic.load_state_dict(torch.load(critic_path))  # 加载critic模型
        checkpoint = np.load(checkpoint_path)  # 加载保存的进度
        return checkpoint['episode'], checkpoint['return_list'].tolist(), checkpoint['case_index']

# PPO的超参数
actor_lr = 1e-3  # 策略网络学习率
critic_lr = 1e-2  # 价值网络学习率
num_episodes = 500  # 总训练回合
gamma = 0.98  # 折扣因子
lmbda = 0.95  # 优势函数平滑系数
epochs = 10  # 每条序列的数据用来训练的轮数
eps = 0.2  # PPO截断范围参数
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 使用GPU或CPU
# 创建数据库连接
db = DB(host='localhost', user='root', password='123456', database='law_data_total')
# 创建对话 Agent 实例
dialogue_agent = Agent()
# 创建自定义的律师嫌疑人对话环境
register(
    id='lawyer_suspector_env_v0',
    entry_point='Lawyer-Suspector-Env.lawyer-suspector-env:LawyerSuspectorEnv',
)
env = gym.make('lawyer_suspector_env_v0', db_connection=db, agent=dialogue_agent,rl_model = 'PPO')
# 设置随机种子
env.seed(0)
torch.manual_seed(0)
# 获取状态空间和动作空间
state_dim = env.observation_space.shape[0]  # 状态空间维度为384
action_dim = env.action_space.n  # 动作空间维度为3
# 创建PPO智能体
checkpoint_dir = "checkpoints_PPO"
ppo_agent = PPO(state_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device,checkpoint_dir)
# 模型和进度保存路径
actor_save_path = os.path.join(checkpoint_dir, "actor_model_PPO.pth")
best_actor_save_path = os.path.join(checkpoint_dir, "actor_model_PPO_best.pth")
critic_save_path = os.path.join(checkpoint_dir, "critic_model_PPO.pth")
best_critic_save_path = os.path.join(checkpoint_dir, "critic_model_PPO_best.pth")
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.npz")
# 确保保存路径存在
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# 恢复模型和训练进度
if os.path.exists(checkpoint_path):
    print("恢复上次训练进度...")
    episode, return_list, case_index = ppo_agent.load_model(actor_save_path, critic_save_path, checkpoint_path)
    print(f"恢复的episode: {episode}, 上次返回值列表的最后值: {return_list[-1]},上次案件索引: {case_index}")
    case_index=case_index+1  # 使用恢复的案件索引重置环境
else:
    print("没有保存的训练进度和模型，开始从头训练...")
    episode, return_list, case_index = 0, [],0
# 初始化最优回报
best_return = float('-inf')  # 初始化为负无穷
# 计算剩余的总回合数
remaining_episodes = num_episodes

# 定期多次评估函数
def evaluate_agent(env, agent, num_episodes=1, case_index=None):
    agent.actor.eval()  # 设置 actor 网络为评估模式
    agent.critic.eval()  # 设置 critic 网络为评估模式
    total_return = 0

    with torch.no_grad():  # 禁用梯度计算
        for _ in range(num_episodes):
            state = env.reset(case_index = case_index)
            done = False
            episode_return = 0
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_return += reward
                state = next_state
            total_return += episode_return
            case_index = case_index + 1

    agent.actor.train()  # 恢复 actor 网络为训练模式
    agent.critic.train()  # 恢复 critic 网络为训练模式
    avg_return = total_return / num_episodes
    return avg_return,case_index

# 定期评估参数
eval_interval = 100 # 每隔##个回合评估一次
num_eval_episodes = 1  # 每次评估##回合数

# 使用tqdm创建进度条，并确保总进度量是更新过的
with tqdm(total=remaining_episodes, initial=episode+1, desc=f"Training or Evaluating Progress") as pbar:
    for i_episode in range(episode+1, num_episodes+1):
        # 定期评估
        if i_episode  % eval_interval == 0:
            print("定期评估")
            avg_eval_return,case_index = evaluate_agent(env, ppo_agent, num_eval_episodes, case_index)
            print(
                f"第 {i_episode} 轮: 在 {num_eval_episodes} 轮评估中，平均回报为 {avg_eval_return:.3f}")
            pbar.update(num_eval_episodes)
            continue

        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset(case_index=case_index)
        done = False
        while not done:
            action = ppo_agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        ppo_agent.update(transition_dict)

        # 保存普通模型
        ppo_agent.save_model(actor_save_path, critic_save_path, return_list, i_episode,case_index)
        print(f"第{i_episode}回合保存模型")

        # 如果当前的episode回报超过了最优回报，保存最优模型
        if episode_return > best_return:
            best_return = episode_return
            ppo_agent.save_best_model(best_actor_save_path, best_critic_save_path)
            print(f"第{i_episode}回合保存最优模型")

        if (i_episode) % 10 == 0:
            pbar.set_postfix({'episode': f'{i_episode}', 'return': f'{np.mean(return_list[-10:]):.3f}'})
            # 绘制阶段性回报曲线，便于观察结果
            episodes_list = list(range(len(return_list)))
            plt.plot(episodes_list, return_list)
            plt.xlabel('Episodes')
            print(return_list)
            plt.ylabel('Returns')
            plt.title('PPO on LawyerSuspectEnv')
            plt.show()
        pbar.update(1)
        case_index = case_index+1
        print(f"此episode的return值为： {episode_return}")

# 绘制回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
print(return_list)
plt.ylabel('Returns')
plt.title('PPO on LawyerSuspectEnv')
plt.show()

# 最后的大规模评估：加载最优模型并评估
print("开始加载最优模型进行大规模评估...")
ppo_agent.load_model(best_actor_save_path, best_critic_save_path, checkpoint_path)

# 执行大规模评估
final_eval_episodes = 50
final_avg_return = evaluate_agent(env, ppo_agent, final_eval_episodes, case_index = 500)
print(f"最终评估: 在 {final_eval_episodes} 轮评估中，平均回报为 {final_avg_return:.3f}")