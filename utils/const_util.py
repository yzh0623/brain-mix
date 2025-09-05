# 激活环境配置
ACTIVATE="local"

# 临时 elasticsearch 索引和字段
TMP_ES_INDEX = "es_gather_qa"
TMP_ES_VECTOR_FIELDS = "gather_vector_1024"

TMP_MYSQL_TURNING_DATA_TABLE = "my_fine_turning_datas"

# 评分开启的线程数
SCORE_THREADS_SIZE = 2

# 获取评分的提示词
def get_score_prompts(qa_content):
    return f"""
        我将提供一条中医药领域的“问答对”（包含问题和回答）。  
        你的任务是：  
        1. 只根据问答对的完整性、准确性、逻辑性和专业性进行质量评估。  
        2. 给出一个 **0 到 10 之间的分数**（10 分表示极高质量，0 分表示极低质量）。  
        3. 只返回一个阿拉伯数字，不要输出任何解释或其他内容。  

        问答对如下：  
        
        {qa_content}

        请直接输出一个 0-10 的整数，不要输出任何解释或符号。
    """
    
def get_question_and_answer_prompts(message):
    
    return f"""

        **输入信息:**
        【内容片段】:{message}

        **任务指令:**
        请根据以上【内容片段】提炼10个独特且多样化的中医药问答对。
        问答内容必须完全基于【内容片段】中的信息。
        问题设计应覆盖：核心理论、方剂药材、诊治方法、性味归经、功效作用、历史背景、适用病症或人群、作用机制或原理、术语解释等。
        
        **输出格式:**
        [
            {{"question": "问题1","answer": "答案1"}},
            {{"question": "问题2","answer": "答案2"}},
            {{"question": "问题3","answer": "答案3"}},
            {{"question": "问题4","answer": "答案4"}},
            {{"question": "问题5","answer": "答案5"}},
            {{"question": "问题6","answer": "答案6"}},
            {{"question": "问题7","answer": "答案7"}},
            {{"question": "问题8","answer": "答案8"}},
            {{"question": "问题9","answer": "答案9"}},
            {{"question": "问题10","answer": "答案10"}}
        ]

    """