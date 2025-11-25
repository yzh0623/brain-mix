"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/knowledges/extractive/auto_choice_params.py
Author: yuanzhenhui
Date: 2025-10-27 00:25:37
LastEditTime: 2025-11-25 09:27:30
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

    def semantic_similarity_score(self, original: str, summary: str) -> float:
        """
        Calculate the semantic similarity score between the original text and the summary.

        This function uses the SentenceTransformer library to calculate the semantic similarity score between the original text and the summary.
        The score is then adjusted based on the ratio of the summary length to the original length.
        If the ratio is not close to 0.5, a penalty is applied to the score.

        Args:
            original (str): The original text.
            summary (str): The summary text.

        Returns:
            float: The semantic similarity score between the original text and the summary.
        """
        if not summary:
            return 0.0
        try:
            # Encode the original and summary texts into embeddings
            emb1 = self.shared_model.encode([original], convert_to_tensor=True)
            emb2 = self.shared_model.encode([summary], convert_to_tensor=True)
            
            # Calculate the semantic similarity score
            sim = float(util.cos_sim(emb1, emb2)[0][0])
            
            # Calculate the ratio of the summary length to the original length
            ratio = len(summary) / max(len(original), 1)
            
            # Apply a penalty if the ratio is not close to 0.5
            penalty = abs(ratio - 0.5)
            score = max(0, sim * (1 - penalty))
            return score
        except Exception as e:
            logger.error(f"semantic similarity calculation failed: {e}")
            return 0.0

    def objective(self, trial, samples):
        """
        This function is the objective function that Optuna will use to evaluate the performance of the model.

        It takes a trial object as an argument and returns a float value representing the performance of the model.

        The function first loads the model and tokenizer using the hyperparameters suggested by Optuna.
        Then, it runs the training session and evaluates the model on the validation set if provided.
        Finally, it returns the evaluation loss or the training loss if the evaluation loss is not available.

        Args:
            trial (optuna.Trial): The trial object that contains the hyperparameters suggested by Optuna.
            samples (list[str]): The list of samples to evaluate the model on.

        Returns:
            float: The evaluation loss or the training loss if the evaluation loss is not available.
        """
        # Define the hyperparameters to be optimized
        params = {
            "compression_ratio": trial.suggest_float("compression_ratio", 0.2, 0.8),
            "lambda_mmr": trial.suggest_float("lambda_mmr", 0.1, 0.9),
            "position_weight": trial.suggest_float("position_weight", 0.0, 0.5),
            "named_entity_weight": trial.suggest_float("named_entity_weight", 0.0, 0.5),
            "number_weight": trial.suggest_float("number_weight", 0.0, 0.5),
            "length_weight": trial.suggest_float("length_weight", 0.0, 0.3),
        }

        # Initialize the ContentCompressor object
        compressor = ContentCompressor(
            sentence_length_limit=400,
            prefilter_ratio=0.7,
            max_sentences=15
        )

        # Initialize the lists to store the scores
        scores = []

        # Initialize the patience, no_improve, and best_score variables
        patience, no_improve, best_score = 5, 0, -np.inf

        # Iterate over the samples and evaluate the model
        for text in tqdm(samples, desc=f"Trial {trial.number}", leave=False, disable=True):
            try:
                score_str = None
                # Compress the text using the hyperparameters
                summary = compressor.compress_text(text, **params)
                
                if not summary or len(summary.strip()) == 0:
                    continue
                
                # Ask the AI to generate code based on human instructions
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
                while True and counter < 3:
                    # Ask the AI to generate code based on human instructions
                    score_str = self.api.chat_with_sync(self.compress_content_param, prompt)
                    if score_str:
                        break
                    else:
                        counter += 1
                    
                if score_str:
                    # Remove the non-digit characters from the score string
                    score_str = re.sub(r'[^\d.]', '', score_str)
                    # Check if the score string is a valid number
                    if score_str and score_str.replace('.', '', 1).isdigit():
                        # Convert the score string to a float
                        llm_score = float(score_str)
                        # Normalize the score to be between 1 and 10
                        llm_score = max(1.0, min(10.0, llm_score))
                        
                        # Calculate the semantic similarity score
                        sss_score = self.semantic_similarity_score(text, summary)
                        
                        # Calculate the final score
                        final_score = sss_score * 5.0 + llm_score * 0.5
                        scores.append(final_score)
                        
                        # Update the best score and no_improve variables
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

        # Check if there are any valid scores
        if not scores:
            logger.warning(f"Trial {trial.number} got no valid scores!")
            return 0.0
        
        # Calculate the average score
        avg_score = np.mean(scores)
        logger.info(f"Trial {trial.number} | Score={avg_score:.4f} | Params={params}")
        return avg_score

    def run_optimization(self, texts, n_trials=50):
        """
        Runs the Optuna parameter optimization.

        Args:
            texts (list[str]): The list of texts to optimize the model on.
            n_trials (int): The number of trials to run. Defaults to 50.

        Returns:
            dict: The best parameters found by the optimization process.
        """
        logger.info("Starting Optuna parameter optimization...")
        logger.info(f"Total samples: {len(texts)}, Trials: {n_trials}")
        
        # Create the Optuna study object
        study = optuna.create_study(direction="maximize")
        
        # Define the objective function
        def objective(trial, texts):
            """
            The objective function to be optimized.

            Args:
                trial (optuna.Trial): The trial object that contains the hyperparameters suggested by Optuna.
                texts (list[str]): The list of texts to optimize the model on.

            Returns:
                float: The evaluation score of the model.
            """
            # Evaluate the model on the validation set
            score = self.objective(trial, texts)
            return score
        
        # Run the optimization
        study.optimize(
            objective, 
            n_trials=n_trials, 
            n_jobs=1,
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