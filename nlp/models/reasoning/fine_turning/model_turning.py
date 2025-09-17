"""
Copyright (c) 2025 by yuanzhenhui All right reserved.
FilePath: /brain-mix/nlp/models/reasoning/fine_turning/model_turning.py
Author: yuanzhenhui
Date: 2025-09-05 17:42:48
LastEditTime: 2025-09-17 23:07:16
"""

import os
import sys
import torch
import gc
import shutil
import time
import json
import random
import numpy as np

from typing import Dict, List, Optional
from itertools import product
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from training_monitor import TrainingMonitor
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class ModelTurning:

    def __init__(self, **kwargs) -> None:
        """
        Initialize the ModelTurning class.
        
        The ModelTurning class is used to fine-tune a pre-trained model using the T4 optimization algorithm.
        
        :param kwargs: Optional keyword arguments for the initialization of the class.
        :return: None
        """
        self.model_cnf_yml = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
        self.model_cnf = YamlUtil(self.model_cnf_yml)
        self.base_model = self.model_cnf.get_value('models.reasoning.origin_model')
        self.turning_path = os.path.join(self.model_cnf.get_value('datasets.train_base_path'), self.model_cnf.get_value('datasets.turning_path'))
        
        # Load the configuration from the YAML file
        self.load_configurations()
        
        # Set up the T4 optimization algorithm
        self.setup_tesla_t4_optimization()
        
        # Initialize the training monitor
        self.training_monitor = TrainingMonitor()

    def load_configurations(self):
        """
        Load the configuration from the YAML file.

        This method loads the configuration from the YAML file and sets up the hyperparameters for the model.

        :return: None
        """
        
        # Load the default hyperparameters
        self.default_params = {
            # Gradient accumulation steps (default: 1)
            'gradient_accumuation_steps': self.model_cnf.get_value('models.reasoning.finetuning.default.gradient_accumuation_steps'),
            
            # Per-device batch size for training (default: 16)
            'per_device_train_batch_size': self.model_cnf.get_value('models.reasoning.finetuning.default.per_device_train_batch_size'),
            
            # Maximum sequence length (default: 512)
            'max_seq_length': self.model_cnf.get_value('models.reasoning.finetuning.default.max_seq_length'),
            
            # Warmup steps (default: 500)
            'warmup_steps': self.model_cnf.get_value('models.reasoning.finetuning.default.warmup_steps'),
            
            # Maximum steps (default: 25000)
            'max_steps': self.model_cnf.get_value('models.reasoning.finetuning.default.max_steps'),
            
            # Learning rate (default: 5e-5)
            'learning_rate': self.model_cnf.get_value('models.reasoning.finetuning.default.learning_rate'),
            
            # Weight decay (default: 0.01)
            'weight_decay': self.model_cnf.get_value('models.reasoning.finetuning.default.weight_decay'),
        }
        
        # Load the dataset size and validation split
        self.dataset_size = self.model_cnf.get_value('models.reasoning.finetuning.dataset_size')
        self.validation_split = self.model_cnf.get_value('models.reasoning.finetuning.validation_split')
        self.data_shuffle = self.model_cnf.get_value('models.reasoning.finetuning.data_shuffle')
        
        # Load the search strategy and max trials
        self.search_strategy = self.model_cnf.get_value('models.reasoning.finetuning.search_strategy')
        self.max_trials = self.model_cnf.get_value('models.reasoning.finetuning.max_trials')
        self.early_stopping_patience = self.model_cnf.get_value('models.reasoning.finetuning.early_stopping_patience')
        
        # Load the search space based on the search strategy
        if self.search_strategy == 'comprehensive':
            self.search_space = self.model_cnf.get_value('models.reasoning.finetuning.performance.comprehensive_search')
        else:
            self.search_space = self.model_cnf.get_value('models.reasoning.finetuning.performance.smart_search')
            
        # Load the target modules
        self.target_modules = self.model_cnf.get_value('models.reasoning.finetuning.performance.target_modules')
        
        # Load the optimization and dataloader configurations
        self.optimization_config = self.model_cnf.get_value('models.reasoning.finetuning.optimization')
        self.dataloader_config = self.model_cnf.get_value('models.reasoning.finetuning.dataloader')

    def setup_tesla_t4_optimization(self):
        """
        Set up the environment for using the Tesla T4 GPU.

        This method sets up the environment for using the Tesla T4 GPU, including setting the CUDA memory allocation configuration and enabling the use of Flash Attention.

        :return: None
        """
        if torch.cuda.is_available():
            # Empty the CUDA cache to free up memory
            torch.cuda.empty_cache()
            
            # Tesla T4 不支持 TF32，必须禁用
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            
            # Benchmark the kernels to find the fastest version
            torch.backends.cudnn.benchmark = True
            
            # Disable the deterministic convolution algorithm
            torch.backends.cudnn.deterministic = False
            
            # Get the properties of the GPU
            gpu_props = torch.cuda.get_device_properties(0)
            
            # Get the name of the GPU
            gpu_name = gpu_props.name
            
            # Get the total memory of the GPU in GB
            gpu_memory = gpu_props.total_memory / 1024**3
            
            # Log the information about the GPU
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"Memory: {gpu_memory:.1f}GB")
            logger.info(f"CUDA capability: {gpu_props.major}.{gpu_props.minor}")
            
            # Tesla T4 specific settings
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            
            # Tesla T4 has compute capability 7.5, supports Flash Attention
            self.use_flash_attention = gpu_props.major >= 7
            if self.use_flash_attention:
                logger.info("Flash Attention available")
            
            # 检测实际支持的精度
            logger.info(f"BF16 support: {torch.cuda.is_bf16_supported()}")  # T4不支持
            logger.info(f"FP16 support: True")  # T4支持FP16

    def load_training_data_from_json(self) -> Optional[DatasetDict]:
        """
        Load the training data from the JSON file.

        This function loads the training data from the JSON file and returns a DatasetDict object.
        If the validation set exists, it will also be loaded.

        :return: A DatasetDict object containing the training and validation sets, or None if an error occurs.
        """
        try:
            # Load the training set from the JSON file
            train_path = os.path.join(self.turning_path, "train_dataset.json")
            with open(train_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            logger.info(f"Load training datasets: {len(train_data)} rows")
            formatted_train_data = self.format_dataset_for_qwen(train_data)
            
            # Load the validation set from the JSON file if it exists
            val_path = os.path.join(self.turning_path, "validation_dataset.json")
            if os.path.exists(val_path):
                with open(val_path, 'r', encoding='utf-8') as f:
                    val_data = json.load(f)
                logger.info(f"Load validation datasets: {len(val_data)} rows")
                formatted_val_data = self.format_dataset_for_qwen(val_data)
                dataset_dict = DatasetDict({'train': Dataset.from_list(formatted_train_data),'validation': Dataset.from_list(formatted_val_data)})
            else:
                dataset_dict = DatasetDict({'train': Dataset.from_list(formatted_train_data)})
            return dataset_dict
        except Exception as e:
            logger.error(f"Load datasets failus: {str(e)}")
            return None

    def get_hyperparameter_combinations(self) -> List[Dict]:
        """
        Get all the hyperparameter combinations.

        If the search strategy is 'smart', return the first `max_trials` combinations.
        If the search strategy is 'comprehensive', return all the combinations by using the `product` function from the `itertools` module.

        :return: A list of dictionaries, where each dictionary represents a hyperparameter combination.
        """
        if self.search_strategy == 'smart':
            # If the search strategy is 'smart', return the first 'max_trials' combinations
            return self.search_space[:self.max_trials]
        else:
            # If the search strategy is 'comprehensive', return all the combinations by using the 'product' function from the 'itertools' module
            keys, values = zip(*self.search_space.items())
            all_combinations = [dict(zip(keys, v)) for v in product(*values)]
            
            if len(all_combinations) > self.max_trials:
                # If there are more combinations than the maximum number of trials, shuffle the combinations and select the first 'max_trials' combinations
                random.shuffle(all_combinations)
                all_combinations = all_combinations[:self.max_trials]
                logger.info(f"Randomly select {self.max_trials} combinations from {len(list(product(*values)))} combinations")
            return all_combinations

    def run_training_with_validation(self, model, tokenizer, dataset_dict, output_dir, hyperparameters):
        """
        Run training with validation.

        This function runs training with validation using the given model, tokenizer, dataset dictionary, output directory, and hyperparameters.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for tokenizing the input data.
            dataset_dict (dict): A dictionary containing the training and validation datasets.
            output_dir (str): The directory where the trained model will be saved.
            hyperparameters (dict): A dictionary containing the hyperparameters for training.

        Returns:
            dict: A dictionary containing the training loss, evaluation loss, training time, and total steps.
        """

        batch_size = int(hyperparameters.get("per_device_train_batch_size", 4))
        grad_accum = int(hyperparameters.get("gradient_accumulation_steps", 4))
        max_steps = int(hyperparameters.get("max_steps", 500))
        warmup_steps = int(self.default_params['warmup_steps'])
        learning_rate = float(hyperparameters.get("learning_rate", 2e-4))
        weight_decay = float(self.default_params['weight_decay'])

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            
            fp16=True,  # T4支持FP16
            bf16=False,  # T4不支持BF16
            
            logging_steps=50,
            save_steps=200,
            eval_steps=200 if 'validation' in dataset_dict else None,
            
            optim="adamw_8bit",  # 使用8bit优化器节省显存
            weight_decay=weight_decay,
            lr_scheduler_type="cosine",  # 简化学习率调度器
            
            seed=42,
            save_total_limit=2,
            load_best_model_at_end=False,
            metric_for_best_model="loss" if 'validation' in dataset_dict else None,
            greater_is_better=False,
            
            dataloader_num_workers=2,  # 减少worker数量避免内存问题
            dataloader_pin_memory=True,
            
            gradient_checkpointing=True,
            group_by_length=True,
            
            # 移除 length_column_name 和 half_precision_backend
            remove_unused_columns=False,
            report_to="none",  # 禁用 wandb 等报告
            push_to_hub=False  # 不推送到 hub
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict.get('validation'),
            dataset_text_field="text",
            max_seq_length=hyperparameters.get("max_seq_length", 2048),
            args=training_args,
            packing=False
        )
        
        start_time = time.time()
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        eval_loss = None
        if 'validation' in dataset_dict:
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get('eval_loss', None)
        
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            'train_loss': train_result.training_loss,
            'eval_loss': eval_loss,
            'training_time': training_time,
            'total_steps': train_result.global_step
        }

    def save_markdown_report(self, report, save_dir):
        """
        Save the training report in markdown format.

        Args:
            report (dict): The training report.
            save_dir (str): The directory to save the report.

        Returns:
            None
        """
        md_path = os.path.join(save_dir, "training_report.md")
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 模型训练报告\n\n")
            
            # Write the generation date
            f.write(f"**生成时间**: {report['training_summary']['date']}\n\n")
            
            # Write the training summary
            f.write("## 训练概况\n\n")
            
            # Write the hardware used for training
            f.write(f"- **硬件**: {report['training_summary']['hardware']}\n")
            
            # Write the base model used for training
            f.write(f"- **基础模型**: {report['training_summary']['base_model']}\n")
            
            # Write the dataset size
            f.write(f"- **数据集大小**: {report['training_summary']['dataset_size']:,}\n")
            
            # Write the total training rounds
            f.write(f"- **训练轮次**: {report['training_summary']['total_rounds']}\n")
            
            # Write the total training time
            f.write(f"- **总训练时间**: {report['training_summary']['total_training_time']}\n\n")
            
            # Write the best model
            f.write("## 最佳模型\n\n")
            
            # Write the round of the best model
            f.write(f"- **轮次**: {report['best_model']['round']}\n")
            
            # Write the training loss of the best model
            f.write(f"- **训练损失**: {report['best_model']['train_loss']:.4f}\n")
            
            # Write the evaluation loss of the best model if exists
            if report['best_model']['eval_loss']:
                f.write(f"- **验证损失**: {report['best_model']['eval_loss']:.4f}\n")
                
            # Write the training time of the best model
            f.write(f"- **训练时间**: {report['best_model']['training_time']}\n\n")
            
            # Write the hyperparameters of the best model
            f.write("### 超参数\n\n")
            
            # Iterate over the hyperparameters and write them to the file
            for key, value in report['best_model']['hyperparameters'].items():
                f.write(f"- **{key}**: {value}\n")

    def save_comprehensive_report(self, all_results, best_model_info, save_dir):
        """
        Save the comprehensive report of the model training process.

        Args:
            all_results (list): A list of dictionaries containing the results of each hyperparameter combination.
            best_model_info (dict): A dictionary containing the information of the best model.
            save_dir (str): The directory to save the report.

        Returns:
            None
        """
        train_losses = [r['train_loss'] for r in all_results]
        eval_losses = [r['eval_loss'] for r in all_results if r['eval_loss'] is not None]
        training_times = [r['training_time'] for r in all_results]
        
        report = {
            "training_summary": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hardware": "Tesla T4",
                "base_model": self.base_model,
                "dataset_size": self.dataset_size,
                "total_rounds": len(all_results),
                "total_training_time": f"{sum(training_times)/3600:.2f} hours",
            },
            "best_model": {
                "round": best_model_info['round'],
                "train_loss": best_model_info['train_loss'],
                "eval_loss": best_model_info.get('eval_loss'),
                "training_time": f"{best_model_info['training_time']/60:.1f} minutes",
                "hyperparameters": best_model_info['hyperparameters']
            },
            "statistics": {
                "train_loss": {
                    "min": min(train_losses),
                    "max": max(train_losses),
                    "mean": np.mean(train_losses),
                    "std": np.std(train_losses)
                },
                "eval_loss": {
                    "min": min(eval_losses) if eval_losses else None,
                    "max": max(eval_losses) if eval_losses else None,
                    "mean": np.mean(eval_losses) if eval_losses else None,
                    "std": np.std(eval_losses) if eval_losses else None
                },
                "training_time": {
                    "min": f"{min(training_times)/60:.1f} min",
                    "max": f"{max(training_times)/60:.1f} min",
                    "mean": f"{np.mean(training_times)/60:.1f} min"
                }
            },
            "all_results": sorted(all_results, key=lambda x: x.get('eval_loss', x['train_loss']))
        }
        
        report_path = os.path.join(save_dir, "training_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.save_markdown_report(report, save_dir)
        logger.info(f"The training report is saved at: {report_path}")

    def format_dataset_for_qwen(self,data_list):
        formatted_data = []
    
        for idx, item in enumerate(data_list):
            try:
                if "messages" in item and isinstance(item["messages"], list) and len(item["messages"]) >= 2:
                    user_content = str(item["messages"][0].get("content", ""))  # 确保是字符串
                    assistant_content = str(item["messages"][1].get("content", ""))  # 确保是字符串
                    
                    # 清理内容
                    user_content = user_content.strip()
                    assistant_content = assistant_content.strip()
                    
                    # 跳过空内容
                    if not user_content or not assistant_content:
                        logger.warning(f"Skipping empty content at index {idx}")
                        continue
                    
                    # Qwen2.5 的对话模板格式
                    formatted_text = f"<|im_start|>system\n你是一位精通中医药学的AI助手。<|im_end|>\n"
                    formatted_text += f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    formatted_text += f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
                    
                    formatted_item = {
                        "text": formatted_text,  # 确保是字符串
                        "input_ids": None,  # 让 trainer 自己处理 tokenization
                        "labels": None,  # 让 trainer 自己处理 labels
                    }
                    
                    formatted_data.append(formatted_item)
                else:
                    logger.warning(f"Invalid data format at index {idx}: {item}")
                    
            except Exception as e:
                logger.error(f"Error processing item at index {idx}: {str(e)}")
                continue
        
        logger.info(f"Formatted {len(formatted_data)} samples for training")
        return formatted_data

    def model_finetuning(self):
        
        # 设置输出目录
        output_dir = os.path.join(self.base_model, self.model_cnf.get_value('models.reasoning.trained_models'))
        final_model_save_dir = os.path.join(self.base_model, self.model_cnf.get_value('models.reasoning.best_model'))
        
        # 清理旧目录
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        logger.info("="*50)
        logger.info("开始加载训练数据...")
        dataset_dict = self.load_training_data_from_json()
        if not dataset_dict:
            logger.error("数据加载失败！")
            return
        
        # 获取超参数组合
        hyperparameter_combinations = self.get_hyperparameter_combinations()
        logger.info(f"将尝试 {len(hyperparameter_combinations)} 种参数组合")
        
        # 初始化结果跟踪
        all_results = []
        best_model_info = None
        best_score = float('inf')
        patience_counter = 0
        
        # 确定数据类型
        dtype = torch.float16
        
        # 开始训练循环
        for round_idx, hyperparameters in enumerate(hyperparameter_combinations, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"轮次 {round_idx}/{len(hyperparameter_combinations)}")
            logger.info(f"超参数: {json.dumps(hyperparameters, indent=2)}")
            
            try:
                # 加载模型
                logger.info("加载模型...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.base_model,
                    max_seq_length=hyperparameters.get('max_seq_length', 2048),
                    dtype=dtype,
                    load_in_4bit=True,  # 使用4bit量化节省显存
                )

                # 修复 tokenizer 的特殊 token
                # Qwen2.5 模型的特殊 token 设置
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

                # 确保 model config 与 tokenizer 一致
                model.config.pad_token_id = tokenizer.pad_token_id
                model.config.bos_token_id = tokenizer.bos_token_id
                model.config.eos_token_id = tokenizer.eos_token_id
                
                # 应用LoRA
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=hyperparameters.get('r', 32),
                    target_modules=self.target_modules,
                    lora_alpha=hyperparameters.get('lora_alpha', 64),
                    lora_dropout=0.1,  # 添加dropout防止过拟合
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=42,
                    use_rslora=True,  # 使用改进的LoRA
                    loftq_config=None,
                )
                
                # 打印模型信息
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
                
                # 训练
                current_model_path = os.path.join(output_dir, f"model_round_{round_idx}")
                results = self.run_training_with_validation(
                    model, tokenizer, dataset_dict, current_model_path, hyperparameters
                )
                
                # 记录结果
                result_info = {
                    'round': round_idx,
                    'path': current_model_path,
                    'hyperparameters': hyperparameters,
                    **results
                }
                all_results.append(result_info)
                
                # 计算综合得分（考虑训练和验证损失）
                if results['eval_loss'] is not None:
                    score = results['eval_loss']  # 优先使用验证损失
                else:
                    score = results['train_loss']
                
                logger.info(f"训练损失: {results['train_loss']:.4f}")
                if results['eval_loss']:
                    logger.info(f"验证损失: {results['eval_loss']:.4f}")
                logger.info(f"训练时间: {results['training_time']/60:.1f} 分钟")
                
                # 更新最佳模型
                if score < best_score:
                    # 删除之前的最佳模型节省空间
                    if best_model_info and os.path.exists(best_model_info['path']):
                        shutil.rmtree(best_model_info['path'])
                    
                    best_score = score
                    best_model_info = result_info
                    patience_counter = 0
                    logger.info("✓ 新的最佳模型！")
                else:
                    # 删除非最佳模型
                    if os.path.exists(current_model_path):
                        shutil.rmtree(current_model_path)
                    patience_counter += 1
                
                # 早停检查
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"连续 {self.early_stopping_patience} 轮无改善，提前停止")
                    break
                
                # 清理内存
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(e)
                logger.error(f"第 {round_idx} 轮训练失败: {str(e)}")
                if "out of memory" in str(e).lower():
                    logger.info("尝试减小batch size或序列长度")
                    # 可以在这里实现自动降级策略
                continue
        
        # 保存最佳模型和报告
        if best_model_info:
            logger.info("\n" + "="*50)
            logger.info("训练完成！最佳模型信息：")
            logger.info(f"训练损失: {best_model_info['train_loss']:.4f}")
            if best_model_info['eval_loss']:
                logger.info(f"验证损失: {best_model_info['eval_loss']:.4f}")
            logger.info(f"训练时间: {best_model_info['training_time']/60:.1f} 分钟")
            logger.info(f"超参数: {json.dumps(best_model_info['hyperparameters'], indent=2)}")
            
            # 复制最佳模型到最终目录
            if os.path.exists(final_model_save_dir):
                shutil.rmtree(final_model_save_dir)
            shutil.copytree(best_model_info['path'], final_model_save_dir)
            
            # 保存详细报告
            self.save_comprehensive_report(all_results, best_model_info, final_model_save_dir)
            
            logger.info(f"\n✓ 最佳模型已保存至: {final_model_save_dir}")
        else:
            logger.error("训练失败，未找到有效模型")

if __name__ == "__main__":
    trainer = ModelTurning()
    trainer.model_finetuning()