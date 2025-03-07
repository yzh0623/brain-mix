"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/pressure_util.py
Author: yuanzhenhui
Date: 2024-11-02 14:30:12
LastEditTime: 2025-01-05 22:14:40
"""

import threading
import multiprocessing as mp
import time
import random
import json
import requests
import signal

import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from yaml_util import YamlConfig
pressure_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'pressure_cnf.yml'))

# gobal question array variable
question_array = []
# global stop event
stop_event = threading.Event()
start_time = None

def load_data_for_test():
    """
    Load data for pressure test from file.

    Load the data from the file specified in the pressure configuration,
    and store it in the `question_array` global variable.
    """
    global question_array
    pressure_file_path = os.path.join(pressure_config, 'resources', 'data','pressure',pressure_config.get_value('pressure.data-filename'))
    with open(pressure_file_path, 'r', encoding='utf-8') as f:
        question_array = json.load(f)

def sse_ask(url, data):
    """
    Send a request to the server and receive a SSE stream.

    The request is sent to the specified `url` with the `data` as the request body.
    The response is expected to be a SSE stream.

    This function is blocking and will return a generator that yields the received
    events.

    Args:
        url (str): The URL to send the request to.
        data (dict): The request body.

    Yields:
        str: The received events.
    """
    headers = {
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json'
    }
    try:
        with requests.post(url, data=json.dumps(data), headers=headers, stream=True) as response:
            if response.status_code == 200:
                buffer = ''
                for line in response.iter_lines(decode_unicode=False):
                    line = line.decode('utf-8')
                    if stop_event.is_set():
                        break
                    if line.startswith('data:'):
                        data = line[5:].strip()
                        if data == '[DONE]':
                            break
                        buffer += data
                    elif line.strip() == '':
                        if buffer:
                            yield buffer
                            buffer = ''
            else:
                logger.error(f"request error, status code:{response.status_code}")
    except Exception as e:
        logger.error(f"SSE request error:{str(e)}")

def sse_totally(queue_id, task, user_id):
    """
    Sends a request to the server and receives a SSE stream for the given task.

    The function checks if the stop event is set, and if not, it sends a request 
    to the server specified in the pressure configuration. It processes the 
    received SSE stream data until the stop event is set or the data transfer is completed.

    Args:
        queue_id (int): The ID of the queue processing the task.
        task (str): The task identifier.
        user_id (int): The ID of the user associated with the task.
    """
    if stop_event.is_set():
        return

    # Retrieve the target URL from the pressure configuration
    url = pressure_config.get_value('pressure.target-url')
    
    # Prepare the request body with user information and a random question
    request_body = {
        "recommend": 0,
        "user_id": user_id,
        "us_id": '',
        "messages": [{"role": 'user', "content": random.choice(question_array)}]
    }

    try:
        # Send the request and process the SSE stream
        for event_data in sse_ask(url, request_body):
            if stop_event.is_set():
                break
            logger.info(f"Queue{queue_id} of {task} receive data:{event_data}")
        logger.info(f"Queue{queue_id} of {task} data transfer completed")
    except Exception as e:
        logger.error(f"task error:{str(e)}")

class TaskHandler(threading.Thread):
    
    def __init__(self, stop_event, queue_id, completion_counter, mode='count'):
        """
        Initialize a task handler that runs in a separate thread.

        Args:
            stop_event (threading.Event): An event that can be set to stop the thread.
            queue_id (int): The ID of the queue that the task handler is running on.
            completion_counter (multiprocessing.Value): A shared counter that keeps track of the number of tasks completed.
            mode (str, optional): The mode of the task handler. 'count' means that the task handler will run a specified number of tasks and then stop, while 'duration' means that the task handler will run for a specified duration and then stop. Defaults to 'count'.
        """
        super().__init__()
        self.stop_event = stop_event
        self.queue_id = queue_id
        self.completion_counter = completion_counter
        self.mode = mode
        self.running = True
        
    def run(self):
        """
        Starts the task handler and runs tasks in a loop until the stop event is set or the specified number of tasks is completed.

        Args:
            None

        Returns:
            None
        """
        try:
            if self.mode == 'count':
                num_tasks = int(pressure_config.get_value('pressure.num-tasks'))
                for i in range(num_tasks):
                    # Check if the stop event is set
                    if self.stop_event.is_set():
                        break
                    # Process a task
                    task_id = f"Thread-{self.queue_id}-Task-{i+1}"
                    self.process_task(task_id)
                    # Increment the completion counter
                    with self.completion_counter.get_lock():
                        self.completion_counter.value += 1
            else:
                # Get the duration from the pressure configuration
                duration = int(pressure_config.get_value('pressure.duration'))
                # Initialize the task counter
                task_counter = 0
                # Run tasks until the duration is reached or the stop event is set
                while (time.time() - start_time) < duration and not self.stop_event.is_set():
                    # Increment the task counter
                    task_counter += 1
                    # Process a task
                    task_id = f"Thread-{self.queue_id}-Task-{task_counter}"
                    self.process_task(task_id)
                    # Increment the completion counter
                    with self.completion_counter.get_lock():
                        self.completion_counter.value += 1
                        
        except Exception as e:
            logger.error(f"Thread{self.queue_id}execute error:{str(e)}")

    def process_task(self, task: str) -> None:
        """
        Processes a task by running a conversation with the server using the SSE protocol.

        Args:
            task (str): The ID of the task to process.

        Returns:
            None
        """
        if self.stop_event.is_set():
            return

        logger.info(f"Thread {self.queue_id} processing {task}, start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        # Get the number of users from the pressure configuration
        user_id = int(pressure_config.get_value('pressure.num-users'))
        # Generate a random user ID
        ran_user_id = random.randint(1, user_id)
        # Run the task by sending a request to the server and receiving a SSE stream
        sse_totally(self.queue_id, task, ran_user_id)
        logger.info(f"Thread {self.queue_id} completed {task}, end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

def signal_handler(signum, frame):
    """
    Handles the SIGINT and SIGTERM signals to stop the threads.

    Args:
        signum (int): The signal number.
        frame (frame): The current stack frame.

    Returns:
        None
    """
    logger.info("receive stop signal，now stop thread...")
    stop_event.set()

def cleanup() -> None:
    """
    Cleans up the resources allocated for the pressure test.

    This function is called when the pressure test is completed or stopped.
    It sets the stop event to stop the threads and waits for 1 second to ensure
    that all the threads are stopped before exiting.
    """
    stop_event.set()
    time.sleep(1)
    logger.info("cleanup complete")

def main():
    """
    The main function of the pressure test script.

    This function reads the configuration from the YAML file, loads the data for the pressure test,
    and starts the specified number of threads to run the tasks. It also handles the SIGINT and
    SIGTERM signals to stop the threads.

    Args:
        None

    Returns:
        None
    """
    global start_time
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Load the data for the pressure test
        load_data_for_test()

        # Get the mode from the pressure configuration
        mode = pressure_config.get_value('pressure.mode')
        # Create a shared counter to keep track of the number of tasks completed
        completion_counter = mp.Value('i', 0)
        # Get the number of threads from the pressure configuration
        num_threads = int(pressure_config.get_value('pressure.num-threads'))
        # Record the start time
        start_time = time.time()
        # Create and start the threads
        threads = []
        for i in range(num_threads):
            handler = TaskHandler(stop_event, i + 1, completion_counter, mode)
            handler.daemon = True
            handler.start()
            threads.append(handler)
        # Wait for all the threads to finish
        for thread in threads:
            thread.join()
        # Calculate the total time taken
        total_time = time.time() - start_time
        logger.info(f"Execute complete used time: {total_time:.2f}s")
        # Log the number of tasks completed and the average QPS
        logger.info(f"Totally complete {completion_counter.value} tasks")
        logger.info(f"Avg QPS: {completion_counter.value/total_time:.2f}")

    except Exception as e:
        # Log any errors that occur
        logger.error(f"main error: {str(e)}")
    finally:
        # Clean up the resources allocated for the pressure test
        cleanup()

if __name__ == "__main__":
    main()