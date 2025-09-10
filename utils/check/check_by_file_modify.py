"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/utils/check/check_by_file_modify.py
Author: Zhenhui Yuan
Date: 2025-09-08 15:11:32
LastEditTime: 2025-09-10 16:11:47
"""

import time
import subprocess
import argparse
from pathlib import Path

import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_dir, 'utils'))

from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

class CheckByFileModify:

    def __init__(self, file_path, script_path, check_interval):
        """
        Initialize the class.

        Args:
            file_path (str): The path to the file to monitor.
            script_path (str): The path to the script to restart.
            check_interval (int): The interval in seconds to check the file modification time.
        """
        self.file_path = Path(file_path)
        self.script_path = Path(script_path)
        self.check_interval = check_interval
        self.process = None
        self.last_modified_time = 0

    def start_script(self):
        """
        Start the script.

        If a process is running, kill it first.
        """
        if self.process:
            self.kill_process()

        logger.info(f"Start script: {self.script_path}")
        try:
            # Use sys.executable to ensure the correct Python interpreter is used
            self.process = subprocess.Popen([
                sys.executable,
                str(self.script_path)
            ])
            logger.info(f"Process started，PID: {self.process.pid}")
            if self.file_path.exists():
                
                # Record the last modified time of the file
                self.last_modified_time = self.file_path.stat().st_mtime
        except Exception as e:
            logger.error(f"Process start error: {e}")
            self.process = None

    def kill_process(self):
        """
        Kill the running process.

        If the process is not running, do nothing.
        """
        if self.process and self.process.poll() is None:
            logger.info(f"Terminate process PID: {self.process.pid}")
            try:
                # Send a terminate signal to the process
                self.process.terminate()
                
                # Wait for the process to exit
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("Process did not exit within 5 seconds, killing it")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"Terminate process error: {e}")
                
        # Reset the process to None
        self.process = None

    def check_file_activity(self):
        """
        Check if the file has been modified since the last check.

        Returns:
            bool: True if the file has been modified, False otherwise.
        """
        try:
            if not self.file_path.exists():
                logger.info(f"File does not exist: {self.file_path}")
                return False

            current_modified_time = self.file_path.stat().st_mtime
            
            # Check if the file has been modified since the last check
            has_activity = current_modified_time > self.last_modified_time
            if has_activity:
                logger.info(f"Detect file activity: {self.file_path}")
                
                # Update the last modified time if the file has been modified
                self.last_modified_time = current_modified_time
            return has_activity
        except Exception as e:
            logger.error(f"Detect file activity error: {e}")
            return False

    def monitor(self):
        """
        Monitor the file modification time and restart the script if the file is modified.

        This method is an infinite loop that checks the file modification time every
        `check_interval` seconds. If the file has been modified, it restarts the script.
        """
        self.start_script()
        try:
            while True:
                # Wait for a certain period of time before checking again
                time.sleep(self.check_interval)

                # Check if the process has exited
                if self.process and self.process.poll() is not None:
                    logger.info(f"Detected process exit code: ({self.process.returncode})")
                    self.start_script()
                    continue

                # Check if the file has been modified
                if not self.check_file_activity():
                    logger.info("Detected no output for a long time, restart the process ..")
                    self.start_script()

        except KeyboardInterrupt:
            logger.error("Received interrupt signal, cleaning in progress ..")
            self.kill_process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor document output and automatically restart processes')
    parser.add_argument('file_path', help='The path of the document to be monitored')
    parser.add_argument('script_path', help='Python script path that needs to be restarted')
    parser.add_argument('check_interval', type=int, help='Check interval time (seconds)')

    args = parser.parse_args()

    if not Path(args.script_path).exists():
        logger.error(f"Error: Script file does not exist- {args.script_path}")
        sys.exit(1)

    monitor = CheckByFileModify(
        file_path=args.file_path,
        check_interval=args.check_interval,
        script_path=args.script_path
    )
    monitor.monitor()
