"""
Copyright (c) 2025 by yuanzhenhui. All right reserved.
FilePath: /brain-mix/utils/llms/cuda_util.py
Author: yuanzhenhui
Date: 2025-01-09 00:06:02
LastEditTime: 2025-01-14 22:06:36
"""
from typing import Dict, Generator,Optional
from multiprocessing import Manager
from pynvml import *
from threading import Lock

import uuid
import time
import torch
import gc
import torch.multiprocessing as mp
import queue

import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))

from yaml_util import YamlConfig
from thrid_util import SiliconUtil
from logging_util import LoggingUtil 
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class CudaMonitor:

    _instance = None
    _initialized = False

    def __init__(self, check_interval: float = 60) -> None:
        """
        Initializes an instance of the CudaMonitor class.

        This constructor initializes GPU monitoring using NVIDIA's NVML.
        It sets up necessary configurations and starts monitoring GPU memory usage.

        Args:
            check_interval (float): The interval in seconds at which the memory 
                                    usage is checked. Defaults to 60 seconds.
        """
        if not CudaMonitor._initialized:
            # Load PyTorch configuration from YAML file
            pytorch_config = YamlConfig(
                os.path.join(project_dir, 'resources', 'config', 'llms', 'pytorch_cnf.yml')
            )
            # Set the GPU memory usage threshold percentage
            self.threshold_percentage = pytorch_config.get_value('pytorch.gpu_memory.threshold_percentage')
            # Set the interval for checking memory usage
            self.check_interval = check_interval
            # Get the current process ID
            self.process_id = os.getpid()
            # Event to control the monitoring thread
            self._stop_monitoring = threading.Event()
            # Thread for monitoring GPU memory usage
            self._monitor_thread: Optional[threading.Thread] = None
            # Initialize the NVIDIA Management Library (NVML)
            nvmlInit()
            # Obtain the NVML handle for the current CUDA device (GPU)
            self.handle = nvmlDeviceGetHandleByIndex(0)
            # Mark this class as initialized
            CudaMonitor._initialized = True

    def __new__(cls, *args, **kwargs):
        """
        Ensure that only one instance of CudaMonitor is created.

        This method is a Singleton pattern implementation. It checks if an
        instance of the class has been created. If not, it creates a new one and
        assigns it to the `_instance` class variable. If an instance already exists,
        it simply returns the existing instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            CudaMonitor: The instance of the class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def is_gpu_available(self,
                         memory_used_threshold: float = 70.0,
                         check_times: int = 3,
                         interval: float = 0.5) -> bool:
        """
        Checks if the GPU is available.

        This method checks if the GPU is available by checking its memory usage.
        If the memory usage exceeds the specified threshold, it will return False.
        Otherwise, it will return True.

        Args:
            memory_used_threshold (float): The threshold for GPU memory usage in percentage. Defaults to 70.0.
            check_times (int): The number of times to check the GPU memory usage. Defaults to 3.
            interval (float): The interval between each check in seconds. Defaults to 0.5.

        Returns:
            bool: True if the GPU is available, False otherwise.
        """
        try:
            for _ in range(check_times):
                # Get the current GPU memory usage
                memory_info = nvmlDeviceGetMemoryInfo(self.handle)
                # Calculate the memory usage percentage
                memory_used_percent = (memory_info.used / memory_info.total) * 100
                # If the memory usage exceeds the threshold, return False
                if memory_used_percent > memory_used_threshold:
                    return False
                # Sleep for the specified interval
                time.sleep(interval)
            # If the GPU is available, return True
            return True
        except Exception:
            # If an exception occurs, return False
            return False

    def _get_process_memory_info(self) -> int:
        """
        Get the current memory usage of the Python process in bytes.

        This method uses the NVIDIA Management Library (NVML) to get the memory
        usage of the Python process. If the process is not using any GPU memory,
        it returns 0.

        Returns:
            int: The current memory usage of the Python process in bytes.
        """
        try:
            # Get a list of all processes running on the GPU
            processes = nvmlDeviceGetComputeRunningProcesses(self.handle)
            # Iterate over the list of processes
            for process in processes:
                # If the current process is found, return its memory usage
                if process.pid == self.process_id:
                    return process.usedGpuMemory
            # If the process is not found, return 0
            return 0
        except:
            # If an exception occurs, return 0
            return 0

    def _get_total_memory(self) -> int:
        """
        Get the total memory of the NVIDIA GPU in bytes.

        This method uses the NVIDIA Management Library (NVML) to get the total
        memory of the NVIDIA GPU.

        Returns:
            int: The total memory of the NVIDIA GPU in bytes.
        """
        # Get the memory info of the device
        info = nvmlDeviceGetMemoryInfo(self.handle)
        # Return the total memory
        return info.total

    def _monitor_memory(self) -> None:
        """
        Monitor the memory usage of the Python process.

        This method monitors the memory usage of the Python process every
        `check_interval` seconds. If the memory usage exceeds the threshold
        percentage, it calls the `clean_memory` method to free up memory.

        Returns:
            None
        """
        while not self._stop_monitoring.is_set():
            # Get the current memory usage of the process
            process_memory = self._get_process_memory_info()
            # Get the total memory of the NVIDIA GPU
            total_memory = self._get_total_memory()
            # If the process is using memory
            if process_memory > 0:
                # Calculate the memory usage percentage
                usage_percentage = (process_memory / total_memory) * 100
                # If the memory usage exceeds the threshold percentage
                if usage_percentage > self.threshold_percentage:
                    # Print a message
                    logger.info(f"Process {self.process_id} memory usage ({usage_percentage:.2f}%) exceeded threshold ({self.threshold_percentage}%)")
                    # Call the clean_memory method to free up memory
                    self.clean_memory()
            # Sleep for the specified interval
            time.sleep(self.check_interval)

    def start_monitoring(self) -> None:
        """
        Start monitoring the memory usage of the Python process.

        This method starts a daemon thread that monitors the memory usage of
        the Python process every `check_interval` seconds. If the memory usage
        exceeds the threshold percentage, it calls the `clean_memory` method to
        free up memory.
        """
        if self._monitor_thread is None:
            # Clear the stop event
            self._stop_monitoring.clear()
            # Create a daemon thread to monitor memory usage
            self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            # Start the thread
            self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """
        Stop monitoring the memory usage of the process.

        This method sets the stop event to signal the monitoring thread to stop.
        It then waits for the thread to finish and sets the thread reference to None.
        """
        if self._monitor_thread is not None:
            # Signal the monitoring thread to stop
            self._stop_monitoring.set()
            # Wait for the monitoring thread to finish
            self._monitor_thread.join()
            # Reset the thread reference
            self._monitor_thread = None

    def clean_memory(self) -> None:
        """
        Free up memory by calling `torch.cuda.empty_cache()` and `gc.collect()`.

        This method is called when the memory usage of the Python process exceeds
        the threshold percentage. It frees up memory by calling
        `torch.cuda.empty_cache()` to release any cached memory on the NVIDIA
        GPU, and then calls `gc.collect()` to free up any unused memory.
        """
        # Release any cached memory on the NVIDIA GPU
        torch.cuda.empty_cache()
        # Free up any unused memory
        gc.collect()

    def __del__(self) -> None:
        """
        Destructor for CudaMonitor class.
        
        This method is automatically called when the instance is about to be destroyed.
        It stops the GPU memory monitoring and shuts down the NVIDIA Management Library (NVML).
        """
        # Stop the memory monitoring thread
        self.stop_monitoring()
        try:
            # Shutdown the NVML
            nvmlShutdown()
        except:
            # Ignore any exceptions during NVML shutdown
            pass

class CudaMultiProcessor:
    
    _instance = None
    _initialized = False
    
    def __init__(self) -> None:
        """
        Initialize the CudaMultiProcessor class.

        This method initializes the CudaMultiProcessor class by setting the
        multiprocessing start method to 'spawn' and creating a multiprocessing pool
        with the specified number of processes. It also creates a Manager() object
        to manage shared state between processes, a dictionary to store response
        queues, and a lock object to synchronize access to the response queues.
        Finally, it creates an instance of the CudaMonitor class to monitor the
        memory usage of the Python process.

        :return: None
        """
        if not CudaMultiProcessor._initialized:
            # Set the multiprocessing start method to 'spawn'
            mp.set_start_method('spawn', force=True)
            pytorch_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'llms', 'pytorch_cnf.yml'))
            # Create a multiprocessing pool with the specified number of processes
            self.pool = mp.Pool(processes = int(pytorch_config.get_value('pytorch.multi_core')))
            # Create a Manager() object to manage shared state between processes
            self.manager = Manager()
            # Create a dictionary to store response queues
            self.response_queues: Dict[str, mp.Queue] = {}
            # Create a lock object to synchronize access to the response queues
            self.lock = Lock()
            # Create an instance of the CudaMonitor class to monitor the memory usage of the Python process
            self.gmm = CudaMonitor()
            self.silicon_instance = SiliconUtil()
            # Set the _initialized flag to True
            CudaMultiProcessor._initialized = True
            
    def __new__(cls, *args, **kwargs):
        """
        Ensure that only one instance of the class is created.

        This method implements the Singleton pattern. It checks if an
        instance of the class has been created. If not, it creates a new
        one and assigns it to the `_instance` class variable. If an
        instance already exists, it simply returns the existing instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The instance of the class.
        """
        if cls._instance is None:
            # Create a new instance
            cls._instance = super().__new__(cls)
        # Return the instance
        return cls._instance

    def start_generation(self, pytorch_instance, need_summary, use_thrid_api: True) -> str:
        """
        Starts the generation process in a separate process.

        Args:
            pytorch_instance: The PyTorch model instance to use for generation.
            need_summary: Whether to generate a summary.
            use_thrid_api: Whether to use the third-party interface computing power.

        Returns:
            The queue ID of the response queue.
        """
        queue_id = str(uuid.uuid4())
        logger.info(f"Starting generation for queue_id: {queue_id}")
        
        response_queue = self.manager.Queue()
        with self.lock:
            # Store the response queue in the dictionary
            self.response_queues[queue_id] = response_queue
            
        if self.gmm.is_gpu_available() == False and use_thrid_api:
            # If GPU resources are scarce, use third-party interface computing power
            logger.debug("GPU resources are scarce, and third-party interface computing power is being enabled...")
            self._silicon_generation_worker(need_summary, response_queue)
        else:
            # Run the generation process in a separate process
            self.pool.apply_async(
                _pytorch_generation_worker,
                args=(pytorch_instance, need_summary, response_queue),
                callback=lambda x: logger.info(f"Process completed for queue_id: {queue_id}"),
                error_callback=lambda e: logger.error(f"Process error for queue_id: {queue_id}: {str(e)}")
            )
        return queue_id
        
    def _silicon_generation_worker(self, need_summary, response_queue) -> None:
        """
        Runs the Silicon Util instance in a separate process to generate text.

        Args:
            need_summary (bool): Whether to generate a summary.
            response_queue (Queue): The queue to put the generated text into.

        Raises:
            Exception: If an error occurs during generation.

        """
        try:
            # Run the Silicon Util instance to generate text
            for chunk in self.silicon_instance.silicon_qwen(need_summary):
                # Put the generated text into the response queue
                response_queue.put(chunk)
        except Exception as e:
            # If an error occurs, put the error message into the response queue
            response_queue.put({"error": e})
        finally:
            # Put None into the response queue to indicate that the generation process has completed
            response_queue.put(None)
    
    def get_results_by_queueid(self, queue_id, timeout=1) -> Generator[dict, None, None]:
        """
        Gets the results for the given queue_id.

        Args:
            queue_id (int): The ID of the queue to get the results from.
            timeout (float, optional): The timeout in seconds to wait for the results. Defaults to 1.

        Yields:
            dict: The result chunk.
        """
        logger.info(f"Getting results for queue_id: {queue_id}")
        try:
            while True:
                try:
                    # Get the result from the response queue
                    result = self.response_queues[queue_id].get(timeout=timeout)
                    if result is None:
                        # If the result is None, it means the generation process has completed
                        break
                    yield result
                except queue.Empty:
                    # If the queue is empty, wait for the next available result
                    continue
        finally:
            with self.lock:
                # Remove the response queue from the dictionary after the results have been retrieved
                if queue_id in self.response_queues:
                    del self.response_queues[queue_id]

    def __del__(self) -> None:
        """
        Closes the multiprocessing pool and waits for all the worker processes to finish.

        This method is called when the instance of the class is garbage collected.
        """
        # Close the multiprocessing pool
        self.pool.close()
        # Wait for all the worker processes to finish
        self.pool.join()
        
def _pytorch_generation_worker(pytorch_instance, need_summary, response_queue) -> None:
    """
    A worker function that is executed in a separate process to generate text using the PyTorch model.

    Args:
        pytorch_instance (PyTorchLLM): The instance of the PyTorch model to use for generation.
        need_summary (bool): Whether to generate a summary.
        response_queue (Queue): The queue to put the generated text into.

    Raises:
        Exception: If an error occurs during generation.

    """
    try:
        for chunk in pytorch_instance.transfor_stream_msg(need_summary):
            response_queue.put(chunk)
    except Exception as e:
        response_queue.put({"error": e})
    finally:
        response_queue.put(None)
