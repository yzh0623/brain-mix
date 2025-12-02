"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/models/reasoning/step2_export_openvino_model.py
Author: yuanzhenhui
Date: 2025-10-10 19:36:15
LastEditTime: 2025-12-01 18:29:30
"""

import shutil
import subprocess
from transformers import AutoModelForCausalLM,AutoTokenizer

from optimum.intel.openvino import OVModelForCausalLM,OVWeightQuantizationConfig
from optimum.exporters.openvino import export_from_model
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_dir, 'utils'))

import const_util as CU
from yaml_util import YamlUtil
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

TASK = "text-generation-with-past"
TRUST_REMOTE_CODE = True

class ExportOpenvinoModel:

    def __init__(self):
        """
        Initialize the ExportOpenvinoModel class.

        The class is used to export an OpenVINO model from a given model directory.
        """
        # Load the model configurations from the YAML file
        self.model_cnf_yml = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
        self.model_cnf = YamlUtil(self.model_cnf_yml)
        
        # The directory where the UnSloth model is located
        self.unsloth_merge_model_dir = self.model_cnf.get_value('models.reasoning.unsloth_merge_model')
        
        # The directory where the OpenVINO model will be saved
        self.openvino_model = self.model_cnf.get_value('models.reasoning.openvino.model')
        shutil.rmtree(self.openvino_model, ignore_errors=True)
        Path(self.openvino_model).mkdir(parents=True, exist_ok=True)
        
        # The directory where the full OpenVINO model will be saved
        self.openvino_full_model_dir = os.path.join(self.openvino_model, "FULL")
        # The directory where the OpenVINO model with INT4 weights will be saved
        self.openvino_int4_model_dir = os.path.join(self.openvino_model, "INT4")
        self.openvino_ovms_model_dir = os.path.join(self.openvino_model, "OVMS")

        # Flags to indicate whether the model has been exported and quantized
        self.exported = False
        self.quant_ok = False
        self.ovms_exported = False
        
        self._load_ovms_config()
        
    def _load_ovms_config(self):
        self.model_name = self.model_cnf.get_value('models.reasoning.openvino.ovms.model_name')
        self.weight_format = self.model_cnf.get_value('models.reasoning.openvino.ovms.weight_format')
        self.enable_prefix_caching = self.model_cnf.get_value('models.reasoning.openvino.ovms.enable_prefix_caching')
        self.cache_size = self.model_cnf.get_value('models.reasoning.openvino.ovms.cache_size')
        self.max_num_seqs = self.model_cnf.get_value('models.reasoning.openvino.ovms.max_num_seqs')
        self.max_num_batched_tokens = self.model_cnf.get_value('models.reasoning.openvino.ovms.max_num_batched_tokens')
        self.reasoning_parser = self.model_cnf.get_value('models.reasoning.openvino.ovms.reasoning_parser')
        self.target_device = self.model_cnf.get_value('models.reasoning.openvino.model_device')

    def openvino_ir_export(self):
        """
        Attempt to export the OpenVINO model from the given UnSloth model directory using the optimum.exporters.openvino.export_from_model() function.

        If the function is not available or fails to import, log an error message and return.
        If the function raises a TypeError, fall back to using the old signature without the **kwargs parameter.
        If the function raises any other exception, log an error message and return.
        If the function is successful, set the self.exported flag to True.
        """
        try:
            # Load the UnSloth model from the given directory
            load_model = AutoModelForCausalLM.from_pretrained(
                self.unsloth_merge_model_dir,
                trust_remote_code=TRUST_REMOTE_CODE
            )
            
            try:
                
                # Use the optimum.exporters.openvino.export_from_model() function to export the OpenVINO model
                logger.info("Using optimum.exporters.openvino.export_from_model(...) to export OpenVINO IR...")
                # Pass the task parameter to the export_from_model() function
                kwargs = {"task": TASK}
                # If TRUST_REMOTE_CODE is True, pass the trust_remote_code parameter to the export_from_model() function
                if TRUST_REMOTE_CODE:
                    kwargs["trust_remote_code"] = True
                    
                # Call the export_from_model() function with the model, output directory, and task parameter
                export_from_model(
                    model=load_model, 
                    output=self.openvino_full_model_dir, 
                    **kwargs
                    )
                
                # Set the self.exported flag to True if the export is successful
                self.exported = True
                logger.info("export_from_model completed.")
            except TypeError:
                # If the export_from_model() function raises a TypeError, fall back to using the old signature without the **kwargs parameter
                logger.info("Falling back to old signature without **kwargs parameter...")
                export_from_model(
                    model=load_model, 
                    output=self.openvino_full_model_dir, 
                    trust_remote_code=TRUST_REMOTE_CODE, 
                    task=TASK
                    )
                
                # Set the self.exported flag to True if the export is successful
                self.exported = True
                logger.info("export_from_model (fallback signature) completed.")
            except Exception as e:
                # If the export_from_model() function raises any other exception, log an error message and return
                logger.error("export_from_model failed:", e)
        except Exception as e:
            # If the export_from_model() function is not available or fails to import, log an error message and return
            logger.error("optimum.exporters.openvino.export_from_model not available or failed to import:", e)

    def quantize_openvino(self):
        """
        Use the OpenVINO Quantizer to quantize the exported model.

        The OpenVINO Quantizer is a Python API that provides a way to quantize
        models without requiring calibration data. The quantization process
        involves converting the model's floating-point weights to integers,
        which reduces the model's precision and memory footprint.

        In this function, we create an instance of the OVQuantizer with
        the default configuration (4-bit integer weights) and pass it to the
        OVModelForCausalLM.from_pretrained() method to load the exported model.
        We then call the save_pretrained() method on the quantized model to save
        it to the specified output directory.

        If the quantization process succeeds, we set the self.quant_ok flag to True.
        If the quantization process fails, we log an error message.
        """
        try:
            quantization_config = OVWeightQuantizationConfig(bits=4)
            openvino_int4_model = OVModelForCausalLM.from_pretrained(
                self.openvino_full_model_dir, 
                quantization_config=quantization_config
                )
            openvino_int4_model.save_pretrained(self.openvino_int4_model_dir)
            self.quant_ok = True
            logger.info("Quantization completed.")
        except Exception as e:
            logger.error("Quantization failed:", e)
    
    def install_ovms_depend(self):
        """
        Install the dependencies required by the OVMS export script.

        The OVMS export script requires the optimum-cli package to be installed.
        This function installs the required package using pip.

        Returns:
            str: The path to the export_model.py script if the installation is successful.
            None: If the installation fails.
        """
        export_script_path = os.path.join(project_dir, 'resources', 'ovms', 'export', 'export_model.py')
        requirements_path = os.path.join(project_dir, 'resources', 'ovms', 'export', 'requirements.txt')
        try:
            logger.info("Installing OVMS export requirements...")
            # Run the pip install command with the requirements.txt file
            subprocess.run(
                ['pip', 'install', '-r', requirements_path],
                check=True
            )
            # Return the path to the export_model.py script if the installation is successful
            return export_script_path
        except Exception as e:
            # Log an error message if the installation fails
            logger.error(f"Failed to download OVMS export script: {e}")
            return None
    
    def export_with_ovms_script(self):
        """
        Export the model using the OpenVINO Model Script (OVMS).

        The OVMS is a Python script provided by OpenVINO that can be used to export
        models from popular deep learning frameworks such as TensorFlow, PyTorch, and
        OpenVINO. This script takes the UnSloth model directory as input and exports the
        model to the OpenVINO model directory.

        Args:
            None

        Returns:
            bool: True if the export is successful, False otherwise.
        """
        export_script_path = self.install_ovms_depend()
        config_path = os.path.join(self.openvino_model, 'config_all.json')

        # Construct the command to run the OVMS export script
        cmd = [
            sys.executable, export_script_path,
            'text_generation',
            '--source_model', self.unsloth_merge_model_dir,
            '--model_repository_path', self.openvino_ovms_model_dir,
            '--model_name', self.model_name,
            '--weight-format', self.weight_format,
            '--target_device', self.target_device,
            '--config_file_path', config_path,
            '--cache_size', str(self.cache_size),
            '--max_num_seqs', str(self.max_num_seqs),
            '--max_num_batched_tokens', str(self.max_num_batched_tokens),
            '--reasoning_parser', self.reasoning_parser,
            '--extra_quantization_params', '--sym --group-size 128'
        ]

        # Add the --enable_prefix_caching flag if prefix caching is enabled
        if self.enable_prefix_caching:
            cmd.append('--enable_prefix_caching')

        try:
            # Run the OVMS export script and capture the output and errors
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Log any warnings or errors from the subprocess
            if result.stderr:
                logger.warning(f"OVMS export stderr: {result.stderr}")
            # Set the self.ovms_exported flag to True if the export is successful
            self.ovms_exported = True
            logger.info("OVMS export completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            # Log any errors from the subprocess
            logger.error(f"OVMS export failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
            
    def has_ir(self, dirpath_str: str) -> bool:
        """
        Check if the given directory path contains the IR files (XML and BIN).

        Args:
            dirpath (Path): The directory path to check.

        Returns:
            bool: True if the directory path contains the IR files, False otherwise.
        """
        # Get the list of XML files
        dirpath = Path(dirpath_str)
        xmls = list(dirpath.rglob("*.xml"))

        # Get the list of BIN files
        bins = list(dirpath.rglob("*.bin"))

        # Return True if both XML and BIN files exist, False otherwise
        return bool(xmls and bins)

    def export_tokenizer(self):
        """
        Export the tokenizer to the OpenVINO model directory.

        This function converts the Hugging Face tokenizer to an OpenVINO tokenizer
        and saves it to the OpenVINO model directory.

        Args:
            None

        Returns:
            None
        """
        from openvino_tokenizers import convert_tokenizer
        from openvino import save_model
        
        # Convert the Hugging Face tokenizer to an OpenVINO tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(
            self.unsloth_merge_model_dir,
            trust_remote_code=True
        )
        ov_tokenizer, ov_detokenizer = convert_tokenizer(
            hf_tokenizer, with_detokenizer=True
        )
        
        # Save the OpenVINO tokenizer to the OpenVINO model directory
        save_model(ov_tokenizer, os.path.join(self.openvino_int4_model_dir, "openvino_tokenizer.xml"))
        save_model(ov_detokenizer, os.path.join(self.openvino_int4_model_dir, "openvino_detokenizer.xml"))
        
        logger.info(f"Tokenizer exported to {self.openvino_int4_model_dir}")

    def start_to_export(self, use_ovms_mode: bool = True):
        """
        Main function to export the model to OpenVINO format.

        This function will first try to export the model using the OVMS script
        if use_ovms_mode is True. If the export fails, it will fall back to
        traditional IR export.

        Args:
            use_ovms_mode (bool): Whether to use the OVMS script to export the model.
                Defaults to True.

        Returns:
            None
        """
        if use_ovms_mode:
            Path(self.openvino_ovms_model_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info("=== Starting OVMS mode export (with reasoning support) ===")
            if self.export_with_ovms_script():
                logger.info("OVMS export completed successfully")
            else:
                logger.error("OVMS export failed, falling back to traditional IR export")
                use_ovms_mode = False
        
        if not use_ovms_mode:
            Path(self.openvino_full_model_dir).mkdir(parents=True, exist_ok=True)
            Path(self.openvino_int4_model_dir).mkdir(parents=True, exist_ok=True)
        
            logger.info("=== Starting traditional IR export ===")
            self.openvino_ir_export()

            if not self.exported:
                logger.error("EXPORT FAILED. Aborting.")
                sys.exit(1)

            if not self.has_ir(self.openvino_full_model_dir):
                logger.error("Warning: export completed but .xml/.bin not found under", self.openvino_full_model_dir)
                logger.error("Proceeding to quantize attempt anyway (some optimum versions create wrapped folders).")
                sys.exit(2)
            
            self.quantize_openvino()

            if not self.quant_ok:
                logger.error("Automatic (no-calib) quantization failed. Please ensure you have 'optimum[intel]' and 'openvino-dev' installed.")
                logger.error("You can still use the exported IR under", self.openvino_full_model_dir)
                sys.exit(3)

            if not self.has_ir(self.openvino_int4_model_dir):
                logger.error("Warning: quantized output dir exists but .xml/.bin not found under", self.openvino_int4_model_dir)
                sys.exit(4)
                
            try:
                self.export_tokenizer()
            except Exception:
                logger.error("FATAL: Tokenizer export failed. Cannot proceed with openvino-genai.")
                sys.exit(5)

        logger.info("Done.")

if __name__ == "__main__":
    eom = ExportOpenvinoModel()
    eom.start_to_export(use_ovms_mode=True)