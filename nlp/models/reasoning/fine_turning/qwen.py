

import warnings

import os
import sys
import torch
import shutil
from itertools import product
from transformers import TrainingArguments
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel


project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

from model_api.poe_util import PoeUtil
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

warnings.filterwarnings("ignore")


class Qwen:

    def __init__(self, **kwargs) -> None:
        self.model_cnf = os.path.join(project_dir, 'resources', 'configs',  'fine_turning.yml')
        self.base_model = YamlUtil(
            self.model_cnf).get_value('models.base_model')
        self.default_gradient_accumulation_steps = YamlUtil(self.model_cnf).get_value(
            'models.finetuning.default.gradient_accumulation_steps')
        self.default_per_device_train_batch_size = YamlUtil(self.model_cnf).get_value(
            'models.finetuning.default.per_device_train_batch_size')
        self.default_max_seq_length = YamlUtil(self.model_cnf).get_value(
            'models.finetuning.default.max_seq_length')

        self.default_warmup_steps = YamlUtil(self.model_cnf).get_value(
            'models.finetuning.default.warmup_steps')
        self.default_max_steps = YamlUtil(self.model_cnf).get_value(
            'models.finetuning.default.max_steps')
        self.default_learning_rate = YamlUtil(self.model_cnf).get_value(
            'models.finetuning.default.learning_rate')
        self.default_weight_decay = YamlUtil(self.model_cnf).get_value(
            'models.finetuning.default.weight_decay')
        self.train_data_dir = YamlUtil(self.model_cnf).get_value(
            'models.finetuning.data_path')

        self.hyperparameter_search_space = {
            "learning_rate": YamlUtil(self.model_cnf).get_value('models.finetuning.performance.learning_rate'),
            "max_steps": YamlUtil(self.model_cnf).get_value('models.finetuning.performance.max_steps'),
            "r": YamlUtil(self.model_cnf).get_value('models.finetuning.performance.r'),
            "lora_alpha": YamlUtil(self.model_cnf).get_value('models.finetuning.performance.lora_alpha'),
            "per_device_train_batch_size": YamlUtil(self.model_cnf).get_value('models.finetuning.performance.per_device_train_batch_size'),
            "gradient_accumulation_steps": YamlUtil(self.model_cnf).get_value('models.finetuning.performance.gradient_accumulation_steps'),
            "max_seq_length": YamlUtil(self.model_cnf).get_value('models.finetuning.performance.max_seq_length')
        }

        self.poe_util = PoeUtil()

    def run_finetuning(self, model, tokenizer, dataset, output_dir, hyperparameters):
        """
        执行一次微调训练。

        参数:
        - model (FastLanguageModel): 要微调的模型。
        - tokenizer (transformers.PreTrainedTokenizer): 模型的tokenizer。
        - dataset (datasets.Dataset): 训练数据集。
        - output_dir (str): 保存模型的目录。
        - hyperparameters (dict): 包含本轮训练超参数的字典。

        返回:
        - float: 本次训练的最终损失。
        """
        gradient_accumulation_steps = hyperparameters.get(
            "gradient_accumulation_steps", self.default_gradient_accumulation_steps)
        per_device_train_batch_size = hyperparameters.get(
            "per_device_train_batch_size", self.default_per_device_train_batch_size)

        trainer = FastLanguageModel.get_finetune_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="messages",
            max_seq_length=hyperparameters.get("max_seq_length", self.default_max_seq_length),

            args=TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=self.default_warmup_steps,
                max_steps=hyperparameters.get("max_steps", self.default_max_steps),
                learning_rate=hyperparameters.get("learning_rate", self.default_learning_rate),
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=self.default_weight_decay,
                lr_scheduler_type="linear",
                seed=42,
                save_total_limit=1,
            )
        )

        trainer_stats = trainer.train()
        return trainer_stats.training_loss

    def model_finetuning(self):

        patience = YamlUtil(self.model_cnf).get_value('models.finetuning.performance.early_stopping')
        output_dir = YamlUtil(self.model_cnf).get_value('models.output_dir')
        final_model_save_dir = YamlUtil(self.model_cnf).get_value('models.save_dir')

        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            logger.error("CUDA不可用。微调需要GPU。", file=sys.stderr)
            return

        try:
            # 使用bf16或fp16进行训练
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            max_seq_lengths = YamlUtil(self.model_cnf).get_value('models.finetuning.performance.max_seq_length')

            # 加载模型和tokenizer，使用4位量化以节省显存
            logger.info("正在加载模型和Tokenizer... (请耐心等待，这可能需要一些时间)")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                max_seq_length=max(max_seq_lengths) if len(max_seq_lengths) > 0 else self.default_max_seq_length,
                dtype=dtype,
                load_in_4bit=True,
            )

            logger.info("模型加载完成。")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("显存不足！请尝试减小 max_seq_length 或 batch_size。", file=sys.stderr)
                return
            else:
                raise e

        # 加载数据集
        dataset = load_dataset("json", data_files=f"{self.train_data_dir}/finetune_data.json", split="train")
        logger.info("数据集加载完成。")

        # --- 自动化寻找最佳模型模式 ---
        logger.info("开始 '自动寻找' 模式...")

        all_models = []
        best_loss = float('inf')
        best_model_path = None
        best_hyperparameters = None
        patience_counter = 0

        # 使用 itertools.product 生成所有超参数组合
        keys, values = zip(*self.hyperparameter_search_space.items())
        hyperparameter_combinations = [dict(zip(keys, v)) for v in product(*values)]

        for round_count, hyperparameters in enumerate(hyperparameter_combinations, 1):
            # 每次循环都重新配置LoRA适配器以应用新的超参数
            logger.info(f"\n--- 正在执行第 {round_count} 轮微调，参数: {hyperparameters} ---")

            try:
                # 应用新的LoRA参数
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=hyperparameters.get("r"),
                    target_modules=YamlUtil(self.model_cnf).get_value('models.finetuning.performance.target_modules'),
                    lora_alpha=hyperparameters.get("lora_alpha"),
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing=True,
                    random_state=42,
                    max_seq_length=hyperparameters.get("max_seq_length"),
                )

                current_model_path = os.path.join(output_dir, f"model_round_{round_count}")
                os.makedirs(current_model_path, exist_ok=True)

                # 执行微调
                current_loss = self.run_finetuning(model, tokenizer, dataset, current_model_path, hyperparameters)
                logger.info(f"第 {round_count} 轮训练损失: {current_loss}")

                # 记录所有训练轮次和模型路径
                all_models.append({
                    "path": current_model_path,
                    "loss": current_loss,
                    "hyperparameters": hyperparameters
                })

                # 判断是否需要早停
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model_path = current_model_path
                    best_hyperparameters = hyperparameters
                    patience_counter = 0  # 损失改善，重置容忍度计数器
                    logger.info(f"当前模型损失 {current_loss} 优于历史最佳损失。继续寻找更优模型...")
                else:
                    patience_counter += 1
                    logger.info(
                        f"当前模型损失 {current_loss} 未能改善。容忍度计数: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        logger.info(f"连续 {patience} 轮损失未改善。自动调优结束。")
                        break

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    logger.error(
                        f"第 {round_count} 轮训练时显存不足。自动调优结束。", file=sys.stderr)
                    break
                else:
                    raise e

        # --- 通过大模型判断最优模型并保存 ---
        if all_models:
            # 构造prompt
            metrics_prompt = "以下是多轮微调的结果，每一项包含模型路径、损失和超参数，请帮我判断哪一轮微调最优，只返回最优模型的路径。\n"
            for m in all_models:
                metrics_prompt += f"模型路径: {m['path']}\n损失: {m['loss']}\n超参数: {m['hyperparameters']}\n\n"
            metrics_prompt += "请只返回最优模型的路径（严格只返回路径字符串）。"

            logger.info("正在通过大模型辅助判断最优模型...")
            best_model_path_from_llm = self.poe_util.poe_check_with_gemini2_5(metrics_prompt)
            if hasattr(best_model_path_from_llm, "strip"):
                best_model_path_from_llm = best_model_path_from_llm.strip()
            logger.info(f"大模型判断的最优模型路径: {best_model_path_from_llm}")

            # 查找最优模型的指标
            best_model = next(
                (m for m in all_models if m["path"] == best_model_path_from_llm), None)
            if best_model:
                best_loss = best_model["loss"]
                best_hyperparameters = best_model["hyperparameters"]
            else:
                logger.error("大模型返回的路径未在已训练模型中找到，默认采用本地最优。")
                best_model_path_from_llm = best_model_path
                best_loss = best_loss
                best_hyperparameters = best_hyperparameters

            logger.info("\n--- 调优完成 ---")
            logger.info(f"最佳模型路径: {best_model_path_from_llm}")
            logger.info(f"最佳损失: {best_loss}")
            logger.info(f"使用的最佳超参数: {best_hyperparameters}")

            logger.info(f"正在将最佳模型保存到: {final_model_save_dir}")
            os.makedirs(final_model_save_dir, exist_ok=True)
            model.save_pretrained(final_model_save_dir)
            tokenizer.save_pretrained(final_model_save_dir)

            # 清理其他微调模型
            logger.info("正在删除其他微调模型...")
            for m in all_models:
                if m["path"] != best_model_path_from_llm:
                    try:
                        shutil.rmtree(m["path"])
                    except Exception as e:
                        logger.error(f"无法删除目录 {m['path']}: {e}")

            logger.info(f"\n最佳微调模型已成功保存至 {final_model_save_dir}。")
            logger.info("您可以使用 `transformers` 库加载并使用这个模型。")
        else:
            logger.error("\n--- 调优失败 ---")
            logger.error("没有找到有效的微调模型。")


# 入口点
if __name__ == "__main__":
    q = Qwen()
    q.model_finetuning()
