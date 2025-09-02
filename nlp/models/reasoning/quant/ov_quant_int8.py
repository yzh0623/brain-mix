from transformers import AutoTokenizer, AutoModel
import torch
import subprocess
import warnings
import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

# 忽略警告
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")


class OVQuantInt8:

    def __init__(self) -> None:

        self.model_cnf = os.path.join(project_dir, 'resources', 'config', 'model_cnf.yml')
        self.base_path = YamlUtil(self.model_cnf).get_value('model.base_path')
        self.quant_path = YamlUtil(self.model_cnf).get_value('model.quant_path')
        self.output_path = os.path.join(project_dir, self.quant_path)

    def ov_quant(self) -> None:
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # 1. 导出为 ONNX
        logger.info("正在导出模型为 ONNX 格式...")
        model = AutoModel.from_pretrained(self.base_path)
        dummy_input = torch.ones(1, 128, dtype=torch.long)
        onnx_path = os.path.join(self.output_path, "ONNX", "qwen3.onnx")
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=17)
        logger.info(f"ONNX 模型已保存到: {onnx_path}")

        # 2. 用 mo.py 转换为 OpenVINO IR
        logger.info("正在使用 mo.py 转换为 OpenVINO IR 格式...")
        ir_output_dir = os.path.join(self.output_path, "IR")
        if not os.path.exists(ir_output_dir):
            os.makedirs(ir_output_dir)
        mo_cmd = [
            "mo",
            "--input_model", onnx_path,
            "--output_dir", ir_output_dir
        ]
        subprocess.run(mo_cmd, check=True)
        logger.info(f"IR 模型已保存到: {ir_output_dir}")

if __name__ == "__main__":
    ov = OVQuantInt8()
    ov.ov_quant()