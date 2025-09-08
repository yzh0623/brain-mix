"""
Copyright (c) 2025 by Zhenhui Yuan All right reserved.
FilePath: /brain-mix/utils/check/check_by_file_modify.py
Author: Zhenhui Yuan
Date: 2025-09-08 15:11:32
LastEditTime: 2025-09-08 15:48:53
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

        logger.info(f"启动脚本: {self.script_path}")
        try:
            # Use sys.executable to ensure the correct Python interpreter is used
            self.process = subprocess.Popen([
                sys.executable,
                str(self.script_path)
            ])
            logger.info(f"进程已启动，PID: {self.process.pid}")

            if self.file_path.exists():
                # Record the last modified time of the file
                self.last_modified_time = self.file_path.stat().st_mtime

        except Exception as e:
            logger.error(f"启动脚本失败: {e}")
            self.process = None

    def kill_process(self):
        """
        Kill the running process.

        If the process is not running, do nothing.
        """
        if self.process and self.process.poll() is None:
            logger.info(f"正在终止进程 PID: {self.process.pid}")
            try:
                # Send a terminate signal to the process
                self.process.terminate()
                # Wait for the process to exit
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If the process does not exit after 5 seconds, kill it
                logger.error("进程未响应，强制终止")
                self.process.kill()
                # Wait for the process to exit
                self.process.wait()
            except Exception as e:
                # Log any errors that occur while killing the process
                logger.error(f"终止进程时出错: {e}")
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
                logger.info(f"文件不存在: {self.file_path}")
                # If the file does not exist, return False
                return False

            current_modified_time = self.file_path.stat().st_mtime
            # Check if the file has been modified since the last check
            has_activity = current_modified_time > self.last_modified_time

            if has_activity:
                logger.info(f"检测到文件更新: {self.file_path}")
                # Update the last modified time if the file has been modified
                self.last_modified_time = current_modified_time

            return has_activity

        except Exception as e:
            logger.error(f"检查文件时出错: {e}")
            # Return False if an error occurs
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
                time.sleep(self.check_interval)

                if self.process and self.process.poll() is not None:
                    logger.info(f"检测到进程已退出 (退出码: {self.process.returncode})")
                    self.start_script()
                    continue

                if not self.check_file_activity():
                    logger.info("检测到长时间无输出，重启进程...")
                    self.start_script()

        except KeyboardInterrupt:
            logger.error("\n收到中断信号，正在清理...")
            self.kill_process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='监控文档输出并自动重启进程')
    parser.add_argument('file_path', help='要监控的文档路径')
    parser.add_argument('script_path', help='需要重启的Python脚本路径')
    parser.add_argument('check_interval', type=int, help='检查间隔时间（秒）')

    args = parser.parse_args()

    if not Path(args.script_path).exists():
        logger.error(f"错误: 脚本文件不存在 - {args.script_path}")
        sys.exit(1)

    monitor = CheckByFileModify(
        file_path=args.file_path,
        check_interval=args.check_interval,
        script_path=args.script_path
    )

    monitor.monitor()
