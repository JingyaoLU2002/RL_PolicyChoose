import pymysql.cursors
import json

# 数据库连接参数
class DB:
    def __init__(self, host, user, password, database):
        self.db_config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }

    def execute_query(self, query, params=None):
        connection = pymysql.connect(**self.db_config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                connection.commit()
                return cursor.fetchall()
        finally:
            connection.close()

    # 检查数据是否存在
    def check_data_exist(self, id, table):
        check_query = f"""
                SELECT count(id)
                FROM {table}
                WHERE id = %s
                """
        check_params = (id,)  # 确保这是一个元组
        result = self.execute_query(check_query, check_params)
        return result[0]["count(id)"]

    # 插入数据
    def insert_record(self, fact, accusation, plaintiff, defendant, extraction_content, classified_info, buli, youli, zhongli, table):
        query = f"""
        INSERT INTO {table} (fact, accusation, plaintiff, defendant, extraction_content, classified_info, buli, youli, zhongli)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (fact, accusation, plaintiff, defendant, extraction_content, classified_info, buli, youli, zhongli)
        self.execute_query(query, params)

    # 更新数据
    def update_all(self, id, fact, extract_content, classified_info, buli, youli, zhongli, table):
        result = self.check_data_exist(id, table)
        if result > 0:
            print("已更新")
            update_query = f"""
                            UPDATE {table}
                            SET fact = %s, extraction_content = %s, classified_info = %s, buli = %s, youli = %s, zhongli = %s
                            WHERE id = %s
                            """
            update_params = (fact, extract_content, classified_info, buli, youli, zhongli, id)
            self.execute_query(update_query, update_params)

    # 更新原告信息
    def update_plaintiff(self, id, plaintiff, table):
        update_query = f"""
                        UPDATE {table}
                        SET id = %s
                        WHERE fact = %s
                        """
        update_params = (id, plaintiff)
        print("已更新")
        self.execute_query(update_query, update_params)

    # 插入或更新数据
    def insert_or_update_record_to_direct(self, id, history, table):
        result = self.check_data_exist(id, table)
        try:
            if result > 0:
                print("已更新")
                # 如果记录存在，则更新记录
                update_query = f"""
                UPDATE {table}
                SET chat_history = %s
                WHERE id = %s
                """
                update_params = (history, id)
                self.execute_query(update_query, update_params)
            else:
                # 如果记录不存在，则插入新记录
                insert_query = f"""
                INSERT INTO {table} (id, chat_history)
                VALUES (%s, %s)
                """
                insert_params = (id, history)
                self.execute_query(insert_query, insert_params)
        except pymysql.MySQLError as e:
            print(e)

    # 删除数据
    def delete_record(self, record_id, table):
        query = f"DELETE FROM {table} WHERE id = %s"
        params = (record_id,)
        self.execute_query(query, params)

    # 查询数据
    def fetch_records(self, column_name, table, condition=None):
        if condition:
            query = f"SELECT {column_name} FROM {table} WHERE {condition}"
        else:
            query = f"SELECT {column_name} FROM {table}"
        return self.execute_query(query)

    # 根据 id 查询数据
    def fetch_record_by_id(self, column_name, table, record_id):
        query = f"SELECT {column_name} FROM {table} WHERE id = %s"
        return self.execute_query(query, (record_id,))

    # 检查字段是否存在
    def check_column_existence(self, table, column_name):
        query = f"""
        SELECT COUNT(*) AS column_count
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = '{self.db_config['database']}'
          AND TABLE_NAME = '{table}'
          AND COLUMN_NAME = '{column_name}'
        """
        result = self.execute_query(query)
        column_count = result[0]['column_count']
        return column_count > 0

    # 增加字段并更新数据
    def add_column_and_update(self, table, new_column_name, values, col_type, id):
        # 增加新字段
        if not self.check_column_existence(table, new_column_name):
            add_column_sql = f"""
               ALTER TABLE {table}
               ADD COLUMN {new_column_name} {col_type};
               """
            self.execute_query(add_column_sql)
        update_column_sql = f"""
           UPDATE {table}
           SET {new_column_name} = %s
           WHERE id = %s;
           """
        self.execute_query(update_column_sql, (values, id))

    # 将对话保存到数据库的指定轮次列中，对于DQN模型
    def save_to_db_DQN(self, case_id, round_number, dialogue):
        column_name = f"round_{round_number}"
        # 将对话内容转换为 JSON 格式字符串
        dialogue_text = {
            "律师": dialogue['律师'],
            "当事人": dialogue['当事人'],
            "策略": dialogue.get('策略', 'N/A')
        }
        dialogue_json = json.dumps(dialogue_text, ensure_ascii=False)  # 转换为 JSON 字符串
        try:
            # 插入或更新语句
            self.execute_query(f"""
                INSERT INTO dialogue_history_DQN (case_id, {column_name})
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE {column_name} = %s
            """, (case_id, dialogue_json, dialogue_json))
        except Exception as e:
            print(f"保存到数据库时出错: {e}")

    # 将对话保存到数据库的指定轮次列中，对于PPO模型
    def save_to_db_PPO(self, case_id, round_number, dialogue):

        column_name = f"round_{round_number}"
        # 将对话内容转换为 JSON 格式字符串
        dialogue_text = {
            "律师": dialogue['律师'],
            "当事人": dialogue['当事人'],
            "策略": dialogue.get('策略', 'N/A')
        }
        dialogue_json = json.dumps(dialogue_text, ensure_ascii=False)  # 转换为 JSON 字符串
        try:
            # 插入或更新语句
            self.execute_query(f"""
                INSERT INTO dialogue_history_PPO (case_id, {column_name})
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE {column_name} = %s
            """, (case_id, dialogue_json, dialogue_json))
        except Exception as e:
            print(f"保存到数据库时出错: {e}")

    # 根据案件ID获取案件索引
    def get_case_index(self, case_id):

        records = self.fetch_records("*", "law_data_total")
        for idx, case in enumerate(records):
            if case["id"] == case_id:
                return idx
        return 0  # 如果未找到，返回0，即从第一个案件开始


