import numpy as np
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from reward_calculator_2 import RewardCalculator
import gym
import json
from gym import spaces

class LawyerSuspectorEnv(gym.Env):

    def __init__(self, db_connection, agent,rl_model):
        """自定义律师与嫌疑人对话环境，基于强化学习的对话策略选择。"""
        super(LawyerSuspectorEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 动作空间：重述、虚张声势、直接询问、选择
        self.sentence_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1') # 使用Sentence BERT模型
        self.db = db_connection  # 数据库连接
        self.agent = agent  # 生成对话的智能体
        self.chat_history = []  # 存储对话历史
        self.current_step = 0  # 跟踪当前对话轮次
        self.max_steps = 10  # 10轮对话为结束条件
        self.case_info = None  # 存储案件信息
        self.case_index = 0  # 跟踪当前案件索引，按顺序获取案件
        self.similarity = [] # 用于计算奖励值的
        self.reward_list = [0] # 用于辅助计算奖励函数的列表
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(768,), dtype=np.float32) # 观测空间为768维向量
        self.reward_calculator = RewardCalculator(agent, db_connection) # RewardCalculator初始化
        self.rl_model = rl_model # 强化学习模型选择

    def reset(self,case_index = 0):
        """重置环境并加载新案件。"""
        self.chat_history.clear()  # 清空对话历史
        self.current_step = 1  # 记录当前是第几步
        self.clear_memory()  # 清除记忆
        self.case_index = case_index # 需要从外部传递，目的是为了满足断电重连的功能
        self.similarity = [] # 清空用于计算奖励值的列表
        self.reward_list = [0]  # 用于辅助计算奖励函数的列表-适用于方法1
        messages = self.db.fetch_records("*", "law_data_total") # 从数据库中获取案件信息
        if not messages:
            raise ValueError("数据库中没有可用案件。") # 确保数据库中有可用案件
        if self.case_index >= len(messages):
            self.case_index = 0  # # 如果超出了案件列表的长度，重置案件索引，确保从第一个案件开始
        case = messages[self.case_index] # 获取案件索引为case_index的案件
        self.case_info = self.process_case(case)  # 提取案件信息
        initial_lawyer_input = "您好，我是律师。请放心，我会尽全力维护您的权益。"
        suspect_output = "您好，我是当事人。我需要您的帮助。" # 初始化对话（律师和当事人初始对话）
        self.chat_history.append({"律师": initial_lawyer_input, "当事人": suspect_output, "策略": "N/A"}) # 把初始化的对话加入对话历史列表中
        print(f"\n轮次: {self.current_step} 策略: N/A")
        print(f"律师: {self.chat_history[-1]['律师']}")
        print(f"当事人: {self.chat_history[-1]['当事人']}")
        if self.rl_model == 'PPO':
            self.db.save_to_db_PPO(self.case_info["id"], 1, self.chat_history[-1])
        else:
            self.db.save_to_db_DQN(self.case_info["id"], 1, self.chat_history[-1]) # 保存第一轮对话到数据库
        return np.zeros(768)  # agent在初始时能够观察到的状态向量

    def process_case(self, case):
        """处理并验证案件信息。"""
        case_info = {
            "fact": case.get("fact", ""), # 从字典对象中提取，get("key", "default_value")
            "buli": case.get("buli", ""),
            "youli": case.get("youli", ""),
            "zhongli": case.get("zhongli", ""),
            "extraction_content": case.get("extraction_content", ""),
            "plaintiff": case.get("plaintiff", ""),
            "defendant": case.get("defendant", ""),
            "id": case.get("id")  # 确保案件ID存在
        }
        if case_info["id"] is None:
            raise ValueError("案件信息缺少ID")  # 如果没有ID，抛出错误
        return case_info

    def step(self, action):
        """根据律师的策略进行对话并更新状态。"""
        if action not in [0, 1, 2, 3]:
            raise ValueError("无效的动作。")
        done = self.current_step >= self.max_steps # 在更新对话之前先判断是否已经达到最大步数
        if done: # 如果到达最大步数，直接结束
            return np.zeros(768), 0, done, {}
        self.current_step += 1
        policy = ["restate", "bluff", "direct","choice"][action]  # 将动作映射到策略
        lawyer_prompt = self.generate_lawyer_prompt(policy) # 生成律师对话提示词
        suspector_prompt = self.generate_suspector_prompt() # 生成嫌疑人对话提示词
        lawyer = self.agent.init_chatbot(lawyer_prompt, "gpt-4o", self.agent.lawyer_memory) # 初始化对话机器人，为律师和传递他的 memory
        suspect = self.agent.init_chatbot(suspector_prompt, "gpt-4o", self.agent.party_memory) # 初始化对话机器人，为嫌疑人传递他的 memory
        suspector_output = self.chat_history[-1]['当事人'] # 获取当前嫌疑人的最后回复或初始值
        lawyer_output = lawyer({"question": suspector_output})["text"] # 获取律师的回复
        suspect_output = suspect({"question": lawyer_output})["text"] # 获取嫌疑人的回复
        self.chat_history.append({"律师": lawyer_output, "当事人": suspect_output, "策略": policy})  # 在对话历史中记录对话以及策略信息
        last_dialog_text = f"律师: {lawyer_output} | 当事人: {suspect_output}" # 获取最后一轮（当前轮）对话的文本
        dialog_vector = self.sentence_model.encode(last_dialog_text) # 将最后一轮（当前轮）对话处理为向量

        # 计算reward值-方法1
        # if self.current_step != 2:
        #     similarities = self.reward_calculator.calculate_reward(self.chat_history, self.case_info["id"])
        #     self.similarity.append(similarities.squeeze(0).tolist())
        #     index_max = np.max(np.array(self.similarity), axis=0)
        #     reward_1 = np.average(index_max)
        #     self.reward_list.append(reward_1)
        #     reward_2 = self.reward_list[-1] - self.reward_list[-2]
        #     reward_3 = reward_2*100
        # else:
        #     similarities = self.reward_calculator.calculate_reward(self.chat_history, self.case_info["id"])
        #     self.similarity.append(similarities.squeeze(0).tolist())
        #     index_max = np.max(np.array(self.similarity), axis=0)
        #     reward_1 = np.average(index_max)
        #     self.reward_list.append(reward_1)
        #     reward_3 = 0

        # 计算reward值-方法2
        if self.current_step != 2:
            # 计算相似度
            similarities = self.reward_calculator.calculate_reward(self.chat_history, self.case_info["id"])
            self.similarity.append(similarities.squeeze(0).tolist())
            # 获取当前 similarity 的最大值
            current_max = np.max(np.array(self.similarity), axis=0)
            if len(self.similarity) > 1:
                # 获取上一轮的最大值
                previous_max = np.max(np.array(self.similarity[:-1]), axis=0)
                # 计算本轮相对于上一轮的提升
                improvement = np.maximum(current_max - previous_max, 0)  # 负值变为0
                # 筛选出提升的部分
                final_improvement = [i for i in improvement if i > 0]  # 只保留正值
                improvement_count = len(final_improvement)  # 统计有提升的项目数量
                # 计算奖励：如果有提升，计算平均值，否则为0
                if improvement_count > 0:
                    reward_1 = sum(final_improvement) / improvement_count  # 只除以提升的项目数量
                else:
                    reward_1 = 0
            else:
                reward_1 = 0  # 如果没有上一轮数据
            self.reward_list.append(reward_1)
            # 计算reward_2和reward_3
            reward_2 = reward_1
            reward_3 = reward_2 * 100
        else:
            # 第二步不计算提升，直接计算 reward
            similarities = self.reward_calculator.calculate_reward(self.chat_history, self.case_info["id"])
            self.similarity.append(similarities.squeeze(0).tolist())
            reward_3 = 0

        # # 将所有对话记录写入 JSON 文件
        # dialog_record = {
        #     "案件ID": self.case_info["id"],
        #     "对话历史": self.chat_history  # 存储完整的对话历史，包括策略
        # }
        # json_filename = f"dialogue/dialog_record_{self.case_info['id']}.json"
        # os.makedirs('dialogue', exist_ok=True) # 新增：确保 dialogue 文件夹存在
        # try: # 打开 JSON 文件以写入完整对话记录，并处理中文字符
        #     with open(json_filename, 'w', encoding='utf-8') as f:
        #         json.dump(dialog_record, f, ensure_ascii=False, indent=4)  # 每次覆盖写入完整记录
        # except Exception as e:
        #     print(f"写入 JSON 文件时出错: {e}")

        # 输出每轮对话和策略
        print(f"\n轮次: {self.current_step} 策略: {policy}")
        print(f"律师: {lawyer_output}")
        print(f"当事人: {suspect_output}")
        print(reward_3)
        # 保存到数据库
        if self.rl_model == 'PPO':
            self.db.save_to_db_PPO(self.case_info["id"], self.current_step, self.chat_history[-1])
        else:
            self.db.save_to_db_DQN(self.case_info["id"], self.current_step, self.chat_history[-1])
        return dialog_vector, reward_3, done, {}

    def get_dialog_history_text(self):
        """将对话历史转换为字符串格式。"""
        return " ".join(
            [f"律师: {entry['律师']} 当事人: {entry['当事人']} 策略: {entry['策略']}" for entry in self.chat_history]
        )

    def generate_suspector_prompt(self):
        """根据案件信息生成嫌疑人的提示词。"""
        return self.agent.read_file("prompt/suspector_prompt.txt").format(
            party=self.case_info["defendant"],
            pla_or_def="被告" if self.case_info["defendant"] else "原告",
            fact=self.case_info["fact"],
            buli=self.case_info["buli"],
            zhongli=self.case_info["zhongli"],
            youli=self.case_info["youli"],
            advance=""
        )

    def generate_lawyer_prompt(self, policy):
        """获取律师的提示词。"""
        with open('prompt/policy_prompt.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        try:
            policy = data["policy"][policy]
        except KeyError as e:
            print(f"参数错误: 缺少字段 {e}")
        with open("prompt/lawyer_prompt.txt", 'r', encoding='utf-8') as file: # 读取文件内容
            prompt = file.read()
        prompt = prompt.format(policy=policy) # 读取文件内容
        return prompt

    def clear_memory(self):
        """清除之前的对话记忆。"""
        self.agent.lawyer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent.party_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def render(self, mode='human'):
        """渲染当前对话历史。"""
        print(self.get_dialog_history_text())
