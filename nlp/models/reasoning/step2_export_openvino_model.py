"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/models/reasoning/step2_export_openvino_model.py
Author: yuanzhenhui
Date: 2025-10-10 19:36:15
LastEditTime: 2025-10-11 17:19:15
"""

import shutil
from transformers import AutoModelForCausalLM

from optimum.intel import OVModelForCausalLM,OVWeightQuantizationConfig
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
        openvino_model = self.model_cnf.get_value('models.reasoning.openvino_model')
        shutil.rmtree(openvino_model, ignore_errors=True)
        Path(openvino_model).mkdir(parents=True, exist_ok=True)
        
        # The directory where the full OpenVINO model will be saved
        self.openvino_full_model_dir = os.path.join(openvino_model, "FULL")
        # The directory where the OpenVINO model with INT4 weights will be saved
        self.openvino_int4_model_dir = os.path.join(openvino_model, "INT4")
        Path(self.openvino_full_model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.openvino_int4_model_dir).mkdir(parents=True, exist_ok=True)

        # Flags to indicate whether the model has been exported and quantized
        self.exported = False
        self.quant_ok = False

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
            openvino_int4_model = OVModelForCausalLM.from_pretrained(self.openvino_full_model_dir, quantization_config=quantization_config)
            openvino_int4_model.save_pretrained(self.openvino_int4_model_dir)
            self.quant_ok = True
            logger.info("Quantization completed.")
        except Exception as e:
            logger.error("Quantization failed:", e)
            
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

    def start_to_export(self):
        """
        Start the export process.

        This function first attempts to export the model using the optimum library.
        If the export fails, it will fall back to using the optimum-cli to export the model.
        If the export is successful, it will then attempt to quantify the exported model using the OpenVINO Quantizer (OVQuantizer).
        If the quantization fails, it will fall back to using a different signature for the quantize method.
        """
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

        logger.info("Done.")

if __name__ == "__main__":
    eom = ExportOpenvinoModel()
    eom.start_to_export()