"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/models/reasoning/step1_model_auto_turning.py
Author: yuanzhenhui
Date: 2025-09-22 10:06:01
LastEditTime: 2025-12-01 13:52:50
"""

import os
import sys
import torch
import gc
import shutil
import time
import json
import optuna
import numpy as np

from typing import Dict, Optional
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel,unsloth_train
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))


class ModelAutoTurning:

    def __init__(self, **kwargs) -> None:
        """
        Initialize the ModelAutoTurning class.

        This class is used to perform automatic turning of the model.

        :param kwargs: The keyword arguments to be passed to the class.
        :type kwargs: Dict[str, Any]
        """
        self.model_cnf_yml = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
        self.model_cnf = YamlUtil(self.model_cnf_yml)
        self.base_model = self.model_cnf.get_value('models.reasoning.origin_model')
        self.turning_path = os.path.join(self.model_cnf.get_value('datasets.train_base_path'), self.model_cnf.get_value('datasets.turning_path'))

        self.load_configurations()
        self.setup_gpu_optimization()

        """
        The dataset dictionary to be used for fine-tuning.
        """
        self.dataset_dict = None
        """
        The output directory to be used for saving the fine-tuned model.
        """
        self.output_dir = None

    def load_configurations(self):
        """
        Load configurations from the YAML file for fine-tuning.

        This function loads the following configurations:

        - dataset_size: The size of the dataset to be used for fine-tuning.
        - max_trials: The maximum number of trials to be performed during fine-tuning.
        - warmup_steps: The number of warmup steps to be performed during fine-tuning.
        - search_space: The search space to be used during fine-tuning, depending on the search strategy.
        - target_modules: The target modules to be fine-tuned.
        - optimization_config: The optimization configuration for fine-tuning.
        - dataloader_config: The dataloader configuration for fine-tuning.

        :return: None
        """
        self.finetuning_config = self.model_cnf.get_value('models.reasoning.finetuning')
        self.max_trials = self.finetuning_config.get('max_trials')
        self.default_params = self.finetuning_config.get('default')
        self.search_space_config = self.finetuning_config.get('performance').get("search_space")
        self.target_modules = self.finetuning_config.get('performance').get("target_modules")
        self.optimization_config = self.finetuning_config.get('optimization')
        self.dataloader_config = self.finetuning_config.get('dataloader')

    def setup_gpu_optimization(self):
        """
        Setup GPU optimization for training.

        This function checks if a CUDA device is available and setups the
        computation type and whether to use bfloat16 for the training.

        :return: None
        """

        # The computation type to be used during training.
        self.compute_dtype = torch.float16
        # Whether to use bfloat16 during training.
        self.use_bf16 = False

        if not torch.cuda.is_available():
            logger.warning("未检测到CUDA设备，将使用CPU进行训练。")
            return

        torch.cuda.empty_cache()

        # Get the properties of the CUDA device.
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"检测到 GPU: {gpu_props.name}, 显存: {gpu_props.total_memory / 1024**3:.1f}GB, CUDA 计算能力: {gpu_props.major}.{gpu_props.minor}")
        if gpu_props.major >= 8:
            logger.info("检测到 Ampere 或更高架构的 GPU。启用 TF32 并优先使用 bfloat16。")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if torch.cuda.is_bf16_supported():
                self.compute_dtype = torch.bfloat16
                self.use_bf16 = True
        torch.backends.cudnn.benchmark = True

    def load_training_data_from_json(self) -> Optional[DatasetDict]:
        """
        Load the training data from the given JSON file.

        This function loads the training data from the given JSON file and
        returns a DatasetDict object containing the training data.

        :return: A DatasetDict object containing the training data, or None if an error occurred.
        """
        try:
            # Load the training data from the JSON file
            train_path = os.path.join(self.turning_path, "train_dataset.json")
            with open(train_path, 'r', encoding='utf-8') as f: train_data = json.load(f)
            logger.info(f"加载训练数据集: {len(train_data)} 条")

            # Load the validation data from the JSON file if it exists
            val_path = os.path.join(self.turning_path, "validation_dataset.json")
            if os.path.exists(val_path):
                with open(val_path, 'r', encoding='utf-8') as f: val_data = json.load(f)
                logger.info(f"加载验证数据集: {len(val_data)} 条")
                dataset_dict = DatasetDict({'train': Dataset.from_list(train_data), 'validation': Dataset.from_list(val_data)})
            else:
                dataset_dict = DatasetDict({'train': Dataset.from_list(train_data)})
            return dataset_dict
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            return None

    def format_dataset_with_chat_template(self, dataset, tokenizer):
        """
        Format the dataset with a chat template.

        This function formats the dataset with a chat template by applying the chat template to
        the messages in the dataset.

        :param dataset: The dataset to format.
        :param tokenizer: The tokenizer to use.
        :return: The formatted dataset.
        """
        def format_prompt(examples):
            """
            Format a single prompt with a chat template.

            :param examples: The prompt to format.
            :return: The formatted prompt.
            """
            return tokenizer.apply_chat_template(examples['messages'], tokenize=False)
        return dataset.map(lambda x: {"text": format_prompt(x)}, num_proc=os.cpu_count() // 2 or 1)

    def run_training_session(self, model, tokenizer, dataset_dict, output_dir, hyperparameters):
        """
        Run a training session with the given model, tokenizer, dataset, output directory, and hyperparameters.

        This function formats the dataset with a chat template, creates a TrainingArguments object with the given hyperparameters,
        creates a SFTTrainer object, trains the model, evaluates the model if a validation dataset is provided,
        saves the model, and returns the training loss, evaluation loss, and training time.

        :param model: The model to train.
        :param tokenizer: The tokenizer to use.
        :param dataset_dict: The dataset to train.
        :param output_dir: The output directory to save the model.
        :param hyperparameters: The hyperparameters to use for training.
        :return: A dictionary containing the training loss, evaluation loss, and training time.
        """
        # Format the dataset with a chat template
        formatted_dataset_dict = DatasetDict()
        formatted_dataset_dict['train'] = self.format_dataset_with_chat_template(dataset_dict['train'], tokenizer)
        if 'validation' in dataset_dict:
            formatted_dataset_dict['validation'] = self.format_dataset_with_chat_template(dataset_dict['validation'], tokenizer)

        # Create a TrainingArguments object with the given hyperparameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=int(hyperparameters.get("per_device_train_batch_size")),
            gradient_accumulation_steps=int(hyperparameters.get("gradient_accumulation_steps")),
            warmup_steps=int(self.default_params.get('warmup_steps', 100)),
            max_steps=int(hyperparameters.get("max_steps")),
            learning_rate=float(hyperparameters.get("learning_rate")),
            fp16=not self.use_bf16,
            bf16=self.use_bf16,
            logging_steps=50,
            save_steps=200,
            eval_steps=200 if 'validation' in formatted_dataset_dict else None,
            optim=self.optimization_config.get("optimizer", "adamw_8bit"),
            lr_scheduler_type=self.optimization_config.get("lr_scheduler_type", "cosine"),
            seed=42,
            save_total_limit=1,
            dataloader_num_workers=self.dataloader_config.get("num_workers", 2),
            gradient_checkpointing=True,
            report_to="none"
        )

        # Create a SFTTrainer object
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=formatted_dataset_dict['train'],
            eval_dataset=formatted_dataset_dict.get('validation'),
            dataset_text_field="text",
            max_seq_length=int(hyperparameters.get("max_seq_length")),
            args=training_args,
            packing=False
        )

        # Train the model
        start_time = time.time()
        trainer = train_on_responses_only(trainer,instruction_part="<|im_start|>user\n",response_part="<|im_start|>assistant\n")
        train_result = unsloth_train(trainer)
        training_time = time.time() - start_time

        # Evaluate the model if a validation dataset is provided
        eval_loss = None
        if 'validation' in formatted_dataset_dict:
            eval_results = trainer.evaluate()
            eval_loss = eval_results.get('eval_loss', None)

        # Save the model
        trainer.save_model(output_dir)

        # Clean up
        del trainer, formatted_dataset_dict
        gc.collect()
        torch.cuda.empty_cache()

        # Return the training loss, evaluation loss, and training time
        return {
            'train_loss': train_result.training_loss,
            'eval_loss': eval_loss,
            'training_time': training_time
        }

    def objective(self, trial: optuna.Trial) -> float:
        """
        This function is the objective function that Optuna will use to evaluate the performance of the model.

        It takes a trial object as an argument and returns a float value representing the performance of the model.

        The function first loads the model and tokenizer using the hyperparameters suggested by Optuna.
        Then, it runs the training session and evaluates the model on the validation set if provided.
        Finally, it returns the evaluation loss or the training loss if the evaluation loss is not available.

        Args:
            trial (optuna.Trial): The trial object that contains the hyperparameters suggested by Optuna.

        Returns:
            float: The evaluation loss or the training loss if the evaluation loss is not available.
        """
        try:
            # Get the hyperparameters suggested by Optuna
            hyperparameters = {
                key: trial.suggest_categorical(key, value['choices'])
                for key, value in self.search_space_config.items()
            }

            # Print the hyperparameters
            logger.info(f"\n{'='*60}\nTrial {trial.number}/{self.max_trials} - Optuna 建议参数:")
            logger.info(json.dumps(hyperparameters, indent=2))

            # Run the training session with the hyperparameters
            current_hyperparameters = hyperparameters.copy()
            while True:
                try:
                    # Load the model and tokenizer
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=self.base_model,
                        max_seq_length=int(current_hyperparameters.get('max_seq_length')),
                        dtype=self.compute_dtype,
                        load_in_4bit=True
                    )

                    # Get the PEFT model
                    model = FastLanguageModel.get_peft_model(
                        model,
                        r=int(current_hyperparameters.get('r')),
                        target_modules=self.target_modules,
                        lora_alpha=int(current_hyperparameters.get('lora_alpha')),
                        lora_dropout=0.0,
                        bias="none",
                        use_gradient_checkpointing="unsloth",
                        random_state=42
                    )

                    # Run the training session
                    current_model_path = os.path.join(self.output_dir, f"trial_{trial.number}")
                    results = self.run_training_session(model, tokenizer, self.dataset_dict, current_model_path, current_hyperparameters)

                    # Get the score (evaluation loss or training loss)
                    score = results.get('eval_loss') if results.get('eval_loss') is not None else results.get('train_loss', float('inf'))

                    # Set the user attributes
                    trial.set_user_attr("model_path", current_model_path)
                    trial.set_user_attr("full_results", {**results, "hyperparameters": current_hyperparameters})

                    # Print the results
                    logger.info(f"Trial {trial.number} 完成。得分 (Loss): {score:.4f}")

                    # Clean up
                    del model, tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()

                    return score

                except torch.cuda.OutOfMemoryError:
                    # Clean up
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Update the hyperparameters
                    current_batch_size = int(current_hyperparameters.get("per_device_train_batch_size"))
                    if current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 2)
                        current_hyperparameters["per_device_train_batch_size"] = new_batch_size
                        logger.warning(f"OOM: 尝试将 batch_size 降低到 {new_batch_size} 后重试。")
                    else:
                        logger.error("OOM: batch_size 已经降低到 1，无法继续运行。")
                        raise optuna.exceptions.TrialPruned()

                except Exception as e:
                    # Print the error message
                    logger.error(f"Trial {trial.number} 发生未知严重错误: {str(e)}", exc_info=True)
                    raise optuna.exceptions.TrialPruned()

        except Exception as e:
            logger.error(f"在 objective 函数中捕获到意外错误: {e}")
            return float('inf')

    def save_comprehensive_report(self, study: optuna.Study, save_dir: str):
        """
        Save the comprehensive report of the hyperparameter optimization study to a JSON file and a Markdown file.

        Args:
            study (optuna.Study): The Optuna study object.
            save_dir (str): The directory where the report will be saved.
        """
        logger.info("正在生成最终的训练报告...")

        # Get all the trial results
        all_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and "full_results" in trial.user_attrs:
                # Get the full results of the trial
                results = trial.user_attrs["full_results"]
                # Get the trial number, evaluation loss, training loss, evaluation loss (if available), and hyperparameters
                result_info = {
                    'trial_number': trial.number,
                    'score': trial.value,
                    'train_loss': results.get('train_loss'),
                    'eval_loss': results.get('eval_loss'),
                    'training_time_seconds': results.get('training_time'),
                    'hyperparameters': results.get('hyperparameters'),
                    'model_path': trial.user_attrs.get("model_path"),
                }
                all_results.append(result_info)

        # Check if there are any successful trials
        if not all_results:
            logger.warning("未找到任何成功的训练试验，无法生成报告。")
            return

        # Get the best trial
        best_trial = study.best_trial
        if not best_trial or "full_results" not in best_trial.user_attrs:
            logger.error("未能找到最佳试验或其结果不完整，报告可能不准确。")
            best_model_info_dict = min(all_results, key=lambda x: x['score'])
        else:
            best_results = best_trial.user_attrs["full_results"]
            best_model_info_dict = {
                'trial_number': best_trial.number,
                'score': best_trial.value,
                'train_loss': best_results.get('train_loss'),
                'eval_loss': best_results.get('eval_loss'),
                'training_time_seconds': best_results.get('training_time'),
                'hyperparameters': best_results.get('hyperparameters'),
            }

        # Get the statistics of the trials
        train_losses = [r['train_loss'] for r in all_results if r.get('train_loss')]
        eval_losses = [r['eval_loss'] for r in all_results if r.get('eval_loss')]
        training_times = [r['training_time_seconds'] for r in all_results if r.get('training_time_seconds')]

        # Generate the report
        report = {
            "training_summary": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_model": self.base_model,
                "total_trials_requested": self.max_trials,
                "total_trials_completed": len(all_results),
                "total_training_time": f"{sum(training_times)/3600:.2f} hours",
            },
            "best_model": best_model_info_dict,
            "statistics": {
                "train_loss": {
                    "min": min(train_losses) if train_losses else None,
                    "max": max(train_losses) if train_losses else None,
                    "mean": np.mean(train_losses) if train_losses else None,
                    "std": np.std(train_losses) if train_losses else None
                },
                "eval_loss": {
                    "min": min(eval_losses) if eval_losses else None,
                    "max": max(eval_losses) if eval_losses else None,
                    "mean": np.mean(eval_losses) if eval_losses else None,
                    "std": np.std(eval_losses) if eval_losses else None
                },
                "training_time_minutes": {
                    "min": f"{min(training_times)/60:.1f}" if training_times else None,
                    "max": f"{max(training_times)/60:.1f}" if training_times else None,
                    "mean": f"{np.mean(training_times)/60:.1f}" if training_times else None
                }
            },
            "all_trials_sorted": sorted(all_results, key=lambda x: x['score'])
        }

        # Save the report to a JSON file
        report_path = os.path.join(save_dir, "training_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save the report to a Markdown file
        self.save_markdown_report(report, save_dir)
        logger.info(f"✓ 详细的训练报告已保存至: {save_dir}")

    def save_markdown_report(self, report: Dict, save_dir: str):
        
        md_path = os.path.join(save_dir, "training_report.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 智能超参数优化训练报告\n\n")

            summary = report['training_summary']
            f.write("## 1. 训练概览\n\n")
            f.write(f"- **生成时间**: {summary['date']}\n")
            f.write(f"- **基础模型**: `{summary['base_model']}`\n")
            f.write(f"- **请求试验次数**: {summary['total_trials_requested']}\n")
            f.write(f"- **成功完成次数**: {summary['total_trials_completed']}\n")
            f.write(f"- **总训练耗时**: {summary['total_training_time']}\n\n")

            best_model = report['best_model']
            f.write("## 2. 最佳模型详情\n\n")
            f.write(f"- **最佳试验编号**: `Trial #{best_model['trial_number']}`\n")
            f.write(f"- **最终得分 (Loss)**: **{best_model['score']:.4f}**\n")
            if best_model.get('eval_loss') is not None: f.write(f"- **验证集损失 (Eval Loss)**: {best_model['eval_loss']:.4f}\n")
            f.write(f"- **训练集损失 (Train Loss)**: {best_model['train_loss']:.4f}\n")
            f.write(f"- **单次训练耗时**: {best_model['training_time_seconds']/60:.1f} 分钟\n\n")

            f.write("### 最佳超参数组合\n\n")
            f.write("```json\n")
            f.write(json.dumps(best_model['hyperparameters'], indent=2, ensure_ascii=False))
            f.write("\n```\n\n")

            stats = report['statistics']
            f.write("## 3. 统计数据摘要\n\n")
            f.write("| 指标 | 最小值 | 最大值 | 平均值 | 标准差 |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|\n")
            if stats['eval_loss']['mean'] is not None:
                f.write(f"| **验证集损失** | {stats['eval_loss']['min']:.4f} | {stats['eval_loss']['max']:.4f} | {stats['eval_loss']['mean']:.4f} | {stats['eval_loss']['std']:.4f} |\n")
            if stats['train_loss']['mean'] is not None:
                f.write(f"| **训练集损失** | {stats['train_loss']['min']:.4f} | {stats['train_loss']['max']:.4f} | {stats['train_loss']['mean']:.4f} | {stats['train_loss']['std']:.4f} |\n")
            if stats['training_time_minutes']['mean'] is not None:
                f.write(f"| **训练耗时(分钟)** | {stats['training_time_minutes']['min']} | {stats['training_time_minutes']['max']} | {stats['training_time_minutes']['mean']} | - |\n")
            f.write("\n")

            f.write("## 4. 所有试验详情 (按性能排序)\n\n")
            header = "| 排名 | Trial # | 得分 (Loss) | 验证集 Loss | 训练集 Loss | 批次大小 | 学习率 | LoRA Rank (r) | 序列长度 |\n"
            separator = "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
            f.write(header)
            f.write(separator)
            for rank, trial_data in enumerate(report['all_trials_sorted'], 1):
                hp = trial_data['hyperparameters']
                eval_loss_str = f"{trial_data['eval_loss']:.4f}" if trial_data.get('eval_loss') is not None else "N/A"
                row = (
                    f"| {rank} "
                    f"| {trial_data['trial_number']} "
                    f"| **{trial_data['score']:.4f}** "
                    f"| {eval_loss_str} "
                    f"| {trial_data['train_loss']:.4f} "
                    f"| {hp['per_device_train_batch_size']} "
                    f"| {hp['learning_rate']} "
                    f"| {hp['r']} "
                    f"| {hp['max_seq_length']} |\n"
                )
                f.write(row)

    def merge_and_save_for_inference(self, best_adapter_path: str, final_save_dir: str) -> None:
        """
        Merge the best LoRA adapter and save the model for inference.

        Args:
            best_adapter_path (str): The path to the best LoRA adapter.
            final_save_dir (str): The directory where the merged model will be saved.
        """
        logger.info("\n" + "="*60)
        logger.info("开始合并模型以用于推理...")

        if os.path.exists(final_save_dir):
            shutil.rmtree(final_save_dir)
        os.makedirs(final_save_dir)

        try:
            # Load the pre-trained model and tokenizer
            trained_model, trained_tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                dtype=self.compute_dtype,
                load_in_4bit=False,
            )

            # Load the best LoRA adapter and merge it with the pre-trained model
            logger.info(f"从 {best_adapter_path} 加载最佳LoRA适配器...")
            model = PeftModel.from_pretrained(trained_model, best_adapter_path)
            
            logger.info("正在合并LoRA权重到基础模型中...")
            model = model.merge_and_unload()

            # Save the merged model to the specified directory
            logger.info(f"正在将合并后的模型保存到: {final_save_dir}")
            model.save_pretrained(final_save_dir)
            trained_tokenizer.save_pretrained(final_save_dir)
            logger.info("✓ 推理模型保存成功！")

        except Exception as e:
            logger.error(f"模型合并与保存过程中发生错误: {str(e)}", exc_info=True)
        finally:
            # Clean up memory
            if 'model' in locals():
                del model
            if 'trained_model' in locals():
                del trained_model
            if 'trained_tokenizer' in locals():
                del trained_tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    def model_finetuning(self):
        """
        Perform model fine-tuning.

        This function loads the training data from a JSON file, creates an Optuna study, optimizes the hyperparameters,
        saves the best model to a directory, generates a comprehensive report, and merges the best LoRA adapter with the pre-trained model for inference.
        """
        # Load the training data from a JSON file
        self.dataset_dict = self.load_training_data_from_json()
        if not self.dataset_dict:
            return

        # Create an Optuna study
        self.output_dir = os.path.join(self.base_model, self.model_cnf.get_value('models.reasoning.trained_models'))
        final_adapter_save_dir = os.path.join(self.base_model, self.model_cnf.get_value('models.reasoning.best_model'))
        final_merged_model_save_dir = f"{final_adapter_save_dir}Merged"

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.max_trials)
        if study.best_trial and study.best_trial.state == optuna.trial.TrialState.COMPLETE:
            best_trial = study.best_trial
            best_model_path = best_trial.user_attrs.get("model_path")

            logger.info("\n" + "="*60 + "\n智能搜索完成！")
            logger.info(f"最佳 Trial: #{best_trial.number}，得分 (Loss): {best_trial.value:.4f}")
            logger.info(f"最佳超参数: {json.dumps(best_trial.params, indent=2)}")

            # Remove the model directories of the non-best trials
            for trial in study.trials:
                if trial.number != best_trial.number and "model_path" in trial.user_attrs:
                    path_to_remove = trial.user_attrs["model_path"]
                    if os.path.exists(path_to_remove):
                        shutil.rmtree(path_to_remove)

            # Save the best LoRA adapter to a directory
            if os.path.exists(final_adapter_save_dir):
                shutil.rmtree(final_adapter_save_dir)
            shutil.copytree(best_model_path, final_adapter_save_dir)

            # Generate a comprehensive report
            self.save_comprehensive_report(study, final_adapter_save_dir)
            logger.info(f"\n✓ 最佳LoRA适配器已保存至: {final_adapter_save_dir}")

            # Merge the best LoRA adapter with the pre-trained model for inference
            self.merge_and_save_for_inference(final_adapter_save_dir, final_merged_model_save_dir)
        else:
            logger.error("训练过程未产生任何有效的最佳模型。")

if __name__ == "__main__":
    trainer = ModelAutoTurning()
    trainer.model_finetuning()