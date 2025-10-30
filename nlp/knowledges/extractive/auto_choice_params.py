"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/knowledges/extractive/auto_choice_params.py
Author: yuanzhenhui
Date: 2025-10-27 00:25:37
LastEditTime: 2025-10-30 11:28:56
"""

import optuna
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import re
import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

from thirdparty.silicon_util import SiliconUtil
from persistence.elastic_util import ElasticUtil
import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from content_compressor import ContentCompressor

es = ElasticUtil()

class AutoChoiceParams:
    
    def __init__(self):
        self.api = SiliconUtil()
        
        self.model_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
        model_cnf = YamlUtil(self.model_cnf)
        self.model_name = model_cnf.get_value('models.compress.path')
        self.device = model_cnf.get_value('models.compress.device')
        
        self.utils_cnf = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'utils_cnf.yml')
        utils_cnf = YamlUtil(self.utils_cnf)
        self.compress_content_param = utils_cnf.get_value('silicon.agent.compress_content')
        
        logger.info(f"Loading model: {self.model_name}")
        self.shared_model = SentenceTransformer(
            model_name_or_path=self.model_name, 
            device=self.device
        )
        logger.info("Model loaded successfully!")

    def semantic_similarity_score(self, original, summary):
        """计算语义相似度分数"""
        # 如果摘要为空，返回0.0
        if not summary:
            return 0.0
        try:
            # 使用共享模型对原始文本进行编码
            emb1 = self.shared_model.encode([original], convert_to_tensor=True)
            # 使用共享模型对摘要文本进行编码
            emb2 = self.shared_model.encode([summary], convert_to_tensor=True)
            # 计算余弦相似度
            sim = float(util.cos_sim(emb1, emb2)[0][0])
            # 计算摘要与原始文本的长度比例
            ratio = len(summary) / max(len(original), 1)
            # 计算长度比例的惩罚值
            penalty = abs(ratio - 0.5)
            # 返回最终的语义相似度分数
            return max(0, sim * (1 - penalty))
        except Exception as e:
            # 捕获异常并记录错误日志
            logger.error(f"Semantic similarity calculation failed: {e}")
            # 返回0.0作为默认值
            return 0.0

    def objective(self, trial, samples):
        """Optuna 目标函数"""
        
        # 调优参数空间
        params = {
            "compression_ratio": trial.suggest_float("compression_ratio", 0.2, 0.8),  # 压缩比例
            "lambda_mmr": trial.suggest_float("lambda_mmr", 0.1, 0.9),  # MMR算法参数
            "position_weight": trial.suggest_float("position_weight", 0.0, 0.5),  # 位置权重
            "named_entity_weight": trial.suggest_float("named_entity_weight", 0.0, 0.5),  # 命名实体权重
            "number_weight": trial.suggest_float("number_weight", 0.0, 0.5),  # 数字权重
            "length_weight": trial.suggest_float("length_weight", 0.0, 0.3),  # 长度权重
        }

        # 初始化内容压缩器
        compressor = ContentCompressor(
            sentence_length_limit=400,  # 句子长度限制
            prefilter_ratio=0.7,  # 预过滤比例
            max_sentences=15  # 最大句子数量
        )

        # 初始化分数列表和早停参数
        scores = []
        patience, no_improve, best_score = 5, 0, -np.inf  # 早停参数

        # 遍历样本
        for text in tqdm(samples, desc=f"Trial {trial.number}", leave=False, disable=True):
            try:
                score_str = None
                # 压缩文本生成摘要
                summary = compressor.compress_text(text, **params)
                
                # 跳过空摘要
                if not summary or len(summary.strip()) == 0:
                    continue
                
                # LLM 评分
                prompt = f"""
                请对以下摘要压缩质量进行评分。

                【原文】
                {text}...

                【摘要】
                {summary}

                请从以下角度综合打分（1~10分）：
                1. 信息保真度
                2. 逻辑连贯性
                3. 压缩效果

                注意：只输出一个数字分数（如 8.5），不要其他内容。
                """
                counter = 0
                # 最多尝试3次获取LLM评分
                while True and counter < 3:
                    score_str = self.api.chat_with_sync(self.compress_content_param, prompt)
                    if score_str:
                        break
                    else:
                        counter += 1
                    
                if score_str:
                    # 清理评分字符串
                    score_str = re.sub(r'[^\d.]', '', score_str)
                    if score_str and score_str.replace('.', '', 1).isdigit():
                        llm_score = float(score_str)
                        # 限制范围
                        llm_score = max(1.0, min(10.0, llm_score))
                        
                        # 计算语义相似度分数
                        sss_score = self.semantic_similarity_score(text, summary)
                        
                        # 计算综合得分
                        final_score = sss_score * 5.0 + llm_score * 0.5  # 归一化到 0-10
                        scores.append(final_score)
                        
                        # 简易早停逻辑
                        if final_score > best_score:
                            best_score = final_score
                            no_improve = 0
                        else:
                            no_improve += 1
                            if no_improve >= patience:
                                break
                            
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {str(e)[:100]}")
                continue

        # 如果没有有效分数，返回0.0
        if not scores:
            logger.warning(f"Trial {trial.number} got no valid scores!")
            return 0.0
        
        # 计算平均分数并记录日志
        avg_score = np.mean(scores)
        logger.info(f"Trial {trial.number} | Score={avg_score:.4f} | Params={params}")
        return avg_score

    def run_optimization(self, texts, n_trials=50):
        logger.info("Starting Optuna parameter optimization...")
        logger.info(f"Total samples: {len(texts)}, Trials: {n_trials}")
        
        study = optuna.create_study(direction="maximize")
        func = lambda trial: self.objective(trial, texts)
        study.optimize(
            func, 
            n_trials=n_trials, 
            n_jobs=1,  # 必须使用单进程
            show_progress_bar=True
        )

        logger.info("\n" + "="*50)
        logger.info("最优参数配置:")
        for k, v in study.best_params.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info(f"最佳得分: {study.best_value:.4f}")
        logger.info("="*50)
        
        return study.best_params

if __name__ == "__main__":
    docs_list = []
    acp = AutoChoiceParams()
    best_params = acp.run_optimization(docs_list, n_trials=10)
    logger.info("\n最终最优参数:")
    logger.info(best_params)