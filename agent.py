import time
import ssl
import warnings
import datetime
import json
import httpx
from openai import OpenAI, APIConnectionError, BadRequestError
from httpx import ConnectError
from collections import Counter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
warnings.filterwarnings("ignore")

class Agent:
    def __init__(self):
        self.current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.party_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.lawyer_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.model_config = {
            "model_name": [
                "glm",
                "gpt",
                "deepseek",
                "moonshot"
            ],
            "config": [
                {
                  "name": "glm",
                  "api_key": "cbbea13466b92475b635d95daf06fd41.MJAnx3BinduYwqKA",

                  "base_url": "https://open.bigmodel.cn/api/paas/v4"
                },
                {
                  "name": "gpt",
                  "api_key": "sk-FjIRXtbymDs4PObfD97b240f4a3a4b9fB66f4dCd28132f43",
                  "base_url": "https://api.rcouyi.com/v1/"
                },
                {
                  "name": "deepseek",
                  "api_key": "sk-6fa9758572754f4882f95e6bddc1fbe0",
                  "base_url": "https://api.deepseek.com"
                },
                {
                  "name": "moonshot",
                  "api_key": "sk-lSc2EpX8GpyPnOjQDEb9WKiQHSzGJ0zWKMwhzyBdzAzJ2bqO",
                  "base_url": "https://api.moonshot.cn/v1"
                }
            ]
        }

    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def write_to_json(self, filename, content):
        with open(filename, 'w', encoding="utf-8") as file:
            json.dump(content, file, indent=4, ensure_ascii=False)

    def find_mode(self, lst):
        if not lst:
            return None  # 如果列表为空，返回 None
        counter = Counter(lst)
        mode, count = counter.most_common(1)[0]  # 获取出现频率最高的元素及其频率
        return mode

    def model(self, prompt, model_name, format=None, n=1, max_retries=10):
        for attempt in range(max_retries):
            try:
                for model in self.model_config["config"]:
                    if model["name"] in model_name.lower():
                        api_key = model["api_key"]
                        base_url = model["base_url"]
                        break
                client = OpenAI(api_key=api_key, base_url=base_url)
                completion = client.chat.completions.create(
                    model=model_name,
                    response_format={"type": format},
                    messages=[{"role": "system", "content": "你是一个有用的智能助手"},
                              {"role": "user", "content": prompt}],
                    top_p=0.5,
                    temperature=0.2,
                    n=n
                )
                all_result = []
                if n == 1:
                    return completion.choices[0].message.content
                else:
                    for item in completion.choices:
                        all_result.append(item.message.content)
                    return all_result
            except (ssl.SSLEOFError,
                    APIConnectionError,
                    ConnectError,
                    ssl.SSLError,
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                    BadRequestError,
                    ) as e:
                print(f"连接错误或其他异常: {e}. 正在重试 {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"连接错误或其他异常: {e}. 正在重试 {attempt + 1}/{max_retries}...")
                print(f"未预见的错误: {e}")

    def init_chatbot(self, template, model_name, memory):
        """初始化对话机器人"""
        for model in self.model_config["config"]:
            if model["name"] in model_name.lower():
                api_key = model["api_key"]
                base_url = model["base_url"]
        llm = ChatOpenAI(temperature=0.6,
                         model=model_name,
                         openai_api_key=api_key,
                         openai_api_base=base_url
                         # base_url="http://10.220.138.110:8000/v1",
                         # api_key="EMPTY",
                         )
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
        prompt = ChatPromptTemplate(
            messages=[system_message_prompt, MessagesPlaceholder(variable_name="chat_history"), human_message_prompt])
        conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)
        return conversation
