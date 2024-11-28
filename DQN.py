import os
import random
import warnings
import numpy as np
import collections
from tqdm import tqdm
from DB import DB
from agent import Agent
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym
from gym.envs.registration import register
warnings.filterwarnings("ignore")


# DQN的Replay Buffer类
class ReplayBuffer:
    '''经验回放池'''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc4 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一个隐藏层使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 第二个隐藏层使用ReLU激活函数
        x = F.relu(self.fc3(x))  # 第三个隐藏层使用ReLU激活函数
        return self.fc4(x)

class DQN:
    def __init__(self, state_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device,checkpoint_dir):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim,self.action_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_dim,self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.checkpoint_dir = checkpoint_dir

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1

    def save_model(self, q_net_path, return_list, episode,case_index):
        # 保存普通模型，并保存episode和return_list
        torch.save(self.q_net.state_dict(), q_net_path)
        np.savez(self.checkpoint_dir + f"/checkpoint.npz", return_list=return_list, episode=episode,case_index = case_index)

    def save_best_model(self, q_net_path):
        # 保存最优模型
        torch.save(self.q_net.state_dict(), q_net_path)
        np.savez(self.checkpoint_dir + f"/checkpoint_best.npz")

    def load_model(self, q_net, checkpoint_path):
        self.q_net.load_state_dict(torch.load(q_net))
        checkpoint = np.load(checkpoint_path)
        return checkpoint['episode'], checkpoint['return_list'].tolist(),checkpoint['case_index']

# DQN超参数
lr = 2e-3
num_episodes = 500 ## Episode总的训练样本
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
db = DB(host='localhost', user='root', password='123456', database='law_data_total')
replay_buffer = ReplayBuffer(buffer_size)
dialogue_agent = Agent()
register(
    id = 'lawyer_suspector_env_v0',
    entry_point='Lawyer-Suspector-Env.lawyer-suspector-env:LawyerSuspectorEnv',
)
env = gym.make('lawyer_suspector_env_v0',db_connection = db, agent = dialogue_agent, rl_model = 'DQN')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
checkpoint_dir = "checkpoints_DQN"
DQN_agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device,checkpoint_dir)
return_list = []
# 模型和进度保存路径
q_net_save_path = os.path.join(checkpoint_dir, "q_net_model_DQN.pth")
best_q_net_save_path = os.path.join(checkpoint_dir, "q_net_model_DQN_best.pth")
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.npz")
# 确保保存路径存在
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# 恢复模型和训练进度
if os.path.exists(checkpoint_path):
    print("恢复上次训练进度...")
    episode, return_list, case_index = DQN_agent.load_model(q_net_save_path, checkpoint_path)
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
    agent.q_net.eval()  # 设置 q_net 网络为评估模式
    total_return = 0
    with torch.no_grad():  # 禁用梯度计算
        for _ in range(num_episodes):
            state = env.reset(case_index=case_index)
            done = False
            episode_return = 0
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_return += reward
                state = next_state
            total_return += episode_return
            case_index = case_index + 1
    agent.q_net.train()  # 恢复 q_net 网络为训练模式
    avg_return = total_return / num_episodes
    return avg_return,case_index
# 定期评估参数
eval_interval = 100  # 每隔##个回合评估一次
num_eval_episodes = 1  # 每次评估##回合数

with tqdm(total=remaining_episodes, initial=episode+1, desc=f"Training or Evaluating Progress") as pbar:
    for i_episode in range(episode+1, num_episodes+1):

        # 定期评估
        if i_episode % eval_interval == 0:
            avg_eval_return,case_index = evaluate_agent(env, DQN_agent, num_eval_episodes, case_index)
            print(
                f"第 {i_episode} 轮: 在 {num_eval_episodes} 轮评估中，平均回报为 {avg_eval_return:.3f}")
            pbar.update(num_eval_episodes)
            continue

        episode_return = 0
        state = env.reset(case_index=case_index)
        done = False
        DQN_agent.hidden_state = None  # 每次重置时也重置隐藏状态
        while not done:
            action = DQN_agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                DQN_agent.update(transition_dict)
        return_list.append(episode_return)
        # 保存普通模型
        DQN_agent.save_model(q_net_save_path, return_list, i_episode,case_index)
        print(f"第{i_episode}回合保存模型")

        # 如果当前的episode回报超过了最优回报，保存最优模型
        if episode_return > best_return:
            best_return = episode_return
            DQN_agent.save_best_model(best_q_net_save_path)
            print(f"第{i_episode}回合保存最优模型")

        if (i_episode) % 10 == 0:
            pbar.set_postfix({'episode': f'{i_episode}', 'return': f'{np.mean(return_list[-10:]):.3f}'})
            # 绘制阶段性回报曲线，便于观察结果
            episodes_list = list(range(len(return_list)))
            plt.plot(episodes_list, return_list)
            plt.xlabel('Episodes')
            print(return_list)
            plt.ylabel('Returns')
            plt.title('DQN on LawyerSuspectEnv')
            plt.show()
        pbar.update(1)
        case_index = case_index + 1
        print(f"此episode的return值为： {episode_return}")

# 绘制回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
print(return_list)
plt.ylabel('Returns')
plt.title('DQN on LawyerSuspectEnv')
plt.show()

# 最后的大规模评估：加载最优模型并评估
print("开始加载最优模型进行大规模评估...")
DQN_agent.load_model(best_q_net_save_path,checkpoint_path)

# 执行大规模评估
final_eval_episodes = 50
final_avg_return = evaluate_agent(env, DQN_agent, final_eval_episodes, case_index = 500)
print(f"最终评估: 在 {final_eval_episodes} 轮评估中，平均回报为 {final_avg_return:.3f}")