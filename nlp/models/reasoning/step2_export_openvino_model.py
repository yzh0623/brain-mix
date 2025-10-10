"""
Copyright (c) 2025 by Zhenhui Yuan. All right reserved.
FilePath: /brain-mix/nlp/models/reasoning/step2_export_openvino_model.py
Author: yuanzhenhui
Date: 2025-10-10 19:36:15
LastEditTime: 2025-10-10 21:12:03
"""


from optimum.intel.openvino.configuration import OVQuantizationConfig
from optimum.intel.openvino import OVQuantizer, OVQuantizer,OVConfig
from optimum.exporters.openvino import export_from_model
import subprocess
from pathlib import Path

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
        self.model_cnf_yml = os.path.join(project_dir, 'resources', 'config', CU.ACTIVATE, 'nlp_cnf.yml')
        unsloth_merge_model = YamlUtil(self.model_cnf_yml).get_value('models.reasoning.unsloth_merge_model')
        openvino_model = YamlUtil(self.model_cnf_yml).get_value('models.reasoning.openvino_model')

        # The directory where the UnSloth model is located
        self.unsloth_merge_model_dir = Path(unsloth_merge_model)
        # The directory where the full OpenVINO model will be saved
        self.openvino_full_model_dir = Path(os.path.join(openvino_model, "FULL"))
        # The directory where the INT4 OpenVINO model will be saved
        self.openvino_int4_model_dir = Path(os.path.join(openvino_model, "INT4"))

        # Create the directories if they do not exist
        self.openvino_full_model_dir.mkdir(parents=True, exist_ok=True)
        self.openvino_int4_model_dir.mkdir(parents=True, exist_ok=True)

        # Flags to indicate whether the model has been exported and quantized
        self.exported = False
        self.quant_ok = False
        
        # OpenVINO configuration
        quant_config = OVQuantizationConfig(
                weights_precision="int4",
                activations_precision="int8",
                preset="performance"
            )
        self.ov_config = OVConfig(quantization_config=quant_config)

    def optimum_export(self):
        """
        Attempt to export the UnSloth model to an OpenVINO model using the optimum library.

        The function first tries to export the model using the export_from_model() function with the task parameter.
        If the function does not exist or fails to import, it will fall back to trying to export the model using the export_from_model() function with the old signature.
        If the fallback also fails, it will log an error message and return.

        Args:
            None

        Returns:
            None
        """
        try:
            try:
                logger.info("Using optimum.exporters.openvino.export_from_model(...) to export OpenVINO IR...")
                kwargs = {"task": TASK}
                if TRUST_REMOTE_CODE:
                    kwargs["trust_remote_code"] = True
                logger.debug(f"export_from_model kwargs: {kwargs}")
                export_from_model(
                    model_or_model_id=str(self.unsloth_merge_model_dir), 
                    output=str(self.openvino_full_model_dir), 
                    **kwargs
                    )
                self.exported = True
                logger.info("export_from_model completed.")
            except TypeError:
                logger.warning("export_from_model failed due to TypeError, falling back to old signature.")
                export_from_model(
                    str(self.unsloth_merge_model_dir), 
                    str(self.openvino_full_model_dir), 
                    TASK
                    )
                self.exported = True
                logger.info("export_from_model (fallback signature) completed.")
            except Exception as e:
                logger.error("export_from_model failed:", e)
        except Exception as e:
            logger.error("optimum.exporters.openvino.export_from_model not available or failed to import:", e)

    def optimum_cli_export(self):
        """
        Attempt to export the UnSloth model to an OpenVINO model using the optimum-cli library.

        The function first tries to export the model using the export_from_model() function with the task parameter.
        If the function does not exist or fails to import, it will fall back to trying to export the model using the export_from_model() function with the old signature.
        If the fallback also fails, it will log an error message and return.

        Args:
            None

        Returns:
            None
        """
        try:
            logger.info("Attempting CLI fallback: 'optimum-cli export' ...")
            cmd = [
                "optimum-cli", "export",
                "--model", str(self.unsloth_merge_model_dir),
                "--format", "openvino",
                "--output", str(self.openvino_full_model_dir),
                "--task", TASK
            ]
            
            # If trust-remote-code is enabled, add the flag to the command
            if TRUST_REMOTE_CODE:
                cmd += ["--trust-remote-code"]
                
            # Run the command
            subprocess.run(cmd, check=True)
            self.exported = True
            logger.info("optimum-cli export completed.")
        except FileNotFoundError:
            logger.error("optimum-cli not found in PATH. Please install optimum and ensure 'optimum-cli' is available, or use Python API.")
        except subprocess.CalledProcessError as e:
            logger.error("optimum-cli export failed. cmd exited with", e.returncode)
        except Exception as e:
            logger.error("optimum-cli export error:", e)

    def quantize_with_ov(self):
        """
        Attempt to quantify the exported OpenVINO model using the OpenVINO Quantizer (OVQuantizer).

        This function creates an instance of the OVQuantizer, sets the model source to the exported OpenVINO model,
        and calls the quantize() method with the output directory set to the INT4 model directory.

        If an exception is raised during the quantization process, it will be logged and the function will return.

        Args:
            None

        Returns:
            None
        """
        try:
            logger.info("Constructing OVQuantizer...")
            q = OVQuantizer(model_src=str(self.openvino_full_model_dir))
            logger.info("Calling q.quantize(output_dir=...) (no calibration data mode)...")
            q.quantize(ov_config=self.ov_config, output_dir=str(self.openvino_int4_model_dir))
            self.quant_ok = True
            logger.info("Quantization completed (no-calib mode).")
        except Exception as e:
            logger.error("OVQuantizer python API failed or quantize() raised an exception:", e)

    def quantize_with_ov_signature_differs(self):
        """
        Attempt to quantify the exported OpenVINO model using the OpenVINO Quantizer (OVQuantizer) with a different signature.

        This function creates an instance of the OVQuantizer, sets the model source to the exported OpenVINO model,
        and calls the quantize() method with the output directory set to the INT4 model directory and subset_size set to 0.

        If an exception is raised during the quantization process, it will be logged and the function will return.

        Args:
            None

        Returns:
            None
        """
        try:
            logger.info("Constructing OVQuantizer...")
            q = OVQuantizer(model_src=str(self.openvino_full_model_dir))
            logger.info("Trying fallback quantize signature: quantize(output_dir=..., subset_size=0)...")
            q.quantize(ov_config=self.ov_config, output_dir=str(self.openvino_int4_model_dir), subset_size=0)
            self.quant_ok = True
            logger.info("Quantization fallback completed.")
        except Exception as e:
            logger.error("Fallback quantize attempt failed:", e)

    def has_ir(self, dirpath: Path) -> bool:
        """
        Check if the given directory path contains the IR files (XML and BIN).

        Args:
            dirpath (Path): The directory path to check.

        Returns:
            bool: True if the directory path contains the IR files, False otherwise.
        """
        # Get the list of XML files
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
        self.optimum_export()
        if not self.exported:
            self.optimum_cli_export()

        if not self.exported:
            logger.error("EXPORT FAILED. Aborting.")
            sys.exit(1)

        if not self.has_ir(self.openvino_full_model_dir):
            logger.error("Warning: export completed but .xml/.bin not found under", self.openvino_full_model_dir)
            logger.error("Proceeding to quantize attempt anyway (some optimum versions create wrapped folders).")

        self.quantize_with_ov()
        if not self.quant_ok:
            self.quantize_with_ov_signature_differs()

        if not self.quant_ok:
            logger.error("Automatic (no-calib) quantization failed. Please ensure you have 'optimum[intel]' and 'openvino-dev' installed.")
            logger.error("You can still use the exported IR under", self.openvino_full_model_dir)
            sys.exit(2)

        if not self.has_ir(self.openvino_int4_model_dir):
            logger.error("Warning: quantized output dir exists but .xml/.bin not found under", self.openvino_int4_model_dir)
        else:
            logger.error("Quantized IR verified under:", self.openvino_int4_model_dir)

        logger.info("Done.")

if __name__ == "__main__":
    eom = ExportOpenvinoModel()
    eom.start_to_export()