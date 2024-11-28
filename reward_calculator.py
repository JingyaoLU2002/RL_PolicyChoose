import json


class RewardCalculator:
    def __init__(self, agent, db):
        self.agent = agent
        self.db = db

    def calculate_reward(self, chat_history, case_id):
        # 构建对话历史
        conversation = ""
        for item in chat_history:
            conversation += f"律师:{item['律师']}\n当事人:{item['当事人']}\n"
        # 根据对话生成预测信息
        prompt_content = self.agent.read_file("prompt/restore_information.txt").format(
            history=conversation, json_formate="{}"
        )
        model_output = self.agent.model(prompt_content, "glm-4-0520", "json_object").strip('```json').strip('```')
        try:
            predict_info = json.loads(model_output)
        except json.JSONDecodeError as e:
            # print("JSON解码错误:", e)
            # print("返回的内容:", model_output)
            predict_info = 0
        # 从数据库获取目标案件信息
        extraction_content = self.db.fetch_record_by_id("extraction_content", "law_data_total", case_id)
        if extraction_content:
            try:
                extract_content = json.loads(extraction_content[0]["extraction_content"])
            except json.JSONDecodeError as e:
                # print("JSON解码错误:", e)
                # print("返回的内容:", extraction_content[0]["extraction_content"])
                return 0
        else:
            return 0
        target_data = []
        if "cases" in extract_content:
            for case in extract_content["cases"]:
                target_data.extend(
                    [f'时间:{case["时间"]}', f'地点:{case["地点"]}', f'人物:{case["人物"]}'] + case["事件"]
                )
        else:
            target_data.extend([
                f'时间:{extract_content["时间"]}',
                f'地点:{extract_content["地点"]}',
                f'人物:{extract_content["人物"]}'
            ] + extract_content["事件"])
        # 准备评估的提示信息
        data_format = '{"完全匹配":[序号],"不匹配":[序号]}'
        prompt_evaluate = self.agent.read_file("prompt/evaluator.txt").format(
            true_info=json.dumps(target_data, ensure_ascii=False),
            predict_info=json.dumps(predict_info, ensure_ascii=False),
            data_format=data_format
        )
        evaluation_output = self.agent.model(prompt_evaluate, "glm-4-0520", "json_object").strip('```json').strip('```')
        try:
            evaluation_results = json.loads(evaluation_output)
        except json.JSONDecodeError as e:
            # print("JSON解码错误:", e)
            # print("返回的内容:", evaluation_output)
            return 0
        # 从评估结果中计数匹配项
        match_count = evaluation_results.get("完全匹配", [])
        if isinstance(match_count, list):
            match_count = len(match_count)
        elif not isinstance(match_count, int):
            match_count = 0
        # 根据匹配数量计算奖励
        reward = match_count

        # # 根据匹配率计算奖励
        # total_info_count = len(target_data) if target_data else 1  # 避免除以零的情况
        # match_rate = match_count / total_info_count  # 计算匹配率
        # reward = match_rate  # 直接使用匹配率作为奖励值

        return reward
