"""
Copyright (c) 2024 by yuanzhenhui All right reserved.
FilePath: /brain-mix/utils/pressure_util.py
Author: yuanzhenhui
Date: 2024-11-02 14:30:12
LastEditTime: 2024-11-08 14:37:09
"""

from yaml_util import YamlConfig

import threading
import multiprocessing
import queue
import time
import random
import json
import requests

import os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 压力配置
pressure_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'pressure_cnf.yml'))
# 队列个数
num_queues = pressure_config.get_value('pressure.num-queues')
mode = pressure_config.get_value('pressure.mode')
limit = pressure_config.get_value('pressure.limit')
interval = pressure_config.get_value('pressure.interval')

# 创建线程数组
threads = []

def loaded_data():
    """
    加载压力测试数据
    """
    global question_array
    pressure_data_path = os.path.join(
        project_dir,
        'resources',
        'data',
        'pressure',
        pressure_config.get_value('pressure.data-filename')
    )
    with open(pressure_data_path, 'r', encoding='utf-8') as f:
        # 遍历获取数组内容
        question_array = json.load(f)


class PressureUtil:
    
    instance = None
    
    def __new__(cls, *args, **kwargs):
        """
        创建PressureUtil的单例实例。

        该静态方法检查类的单例实例是否已存在。
        如果不存在，则创建一个新的实例并返回。
        通过单例模式确保只有一个PressureUtil实例被创建。

        参数:
            *args: 任意位置参数。
            **kwargs: 任意关键字参数。

        返回:
            PressureUtil: PressureUtil的单例实例。
        """
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def sse_ask(self, url, data):
        """
        使用给定的数据向指定的URL发送POST请求，期望得到服务器发送事件（SSE）响应。

        此函数在数据到达时从服务器流式传输数据，生成每个完整的事件。
        它逐行处理响应，收集数据，直到遇到空白行，
        这表示事件的结束。事件由以“data:”开头的行标识。
        当函数遇到“[DONE]”或响应不成功时，它将停止处理。

        参数：
            url（str）：POST请求发送到的url。
            data（dict）：请求体中要发送的数据，编码为JSON。
        """
        headers = {
            'Accept': 'text/event-stream',
            'Content-Type': 'application/json'
        }
        response = requests.post(url, json=data, headers=headers, stream=True)
        if response.status_code != 200:
            raise Exception(f"请求失败，状态码：{response.status_code}")

        buffer = []
        for line in response.iter_lines(decode_unicode=False):
            line = line.decode('utf-8')
            if line.startswith('data:'):
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                buffer.append(data)
            elif not line.strip() and buffer:
                yield ''.join(buffer)
                buffer.clear()

    def sse_totally(self, queue_id, task, user_id):
        """
        向指定 URL 发送服务器发送事件 (SSE) 请求，并处理响应。
        此函数构造一个包含用户ID和随机选择的问题的请求体
        从预加载的列表中选择。它将请求发送到配置文件中指定的URL
        并对接收的事件数据进行迭代，将每个事件打印到控制台。一旦所有
        当事件被处理时，它会打印一条完成消息。

        参数：
            user_id (str): 发出请求的用户的 ID。
        """
        url = pressure_config.get_value('pressure.target-url')
        request_body = {
            "recommend": 0,
            "user_id": user_id,
            "us_id": '',
            "messages": [
                {
                    "role": 'user',
                    "content": random.choice(question_array)
                }
            ]
        }
        for event_data in self.sse_ask(url, request_body):
            print(f"Queue{queue_id}的{task}接收到事件数据：{event_data}")
        print(f"Queue{queue_id}的{task}数据传输已完成")


class TaskHandler(threading.Thread):

    def __init__(self, q, stop_event, queue_id, completion_counter):
        """
        初始化一个TaskHandler实例。

        参数：
            q (queue.Queue): 从其中获取任务的队列。
            stop_event (threading.Event): 一个用于指示何时停止处理任务的事件。
            queue_id (int)：队列的标识符。
            completion_counter (dict): 跟踪已完成任务数量的计数器。

        属性：
            running (bool): 任务处理程序当前是否正在运行的标志。
        """
        super().__init__()
        self.queue = q
        self.stop_event = stop_event
        self.queue_id = queue_id
        self.completion_counter = completion_counter
        self.running = True

    def run(self):
        """
        任务处理程序主循环。
        在这个循环中，我们会不断地从队列中获取任务，并将其交由 process_task() 方法来处理。如果停止信号被设置，我们会清空队列并退出循环。
        在处理每个任务时，我们会在完成任务后将其从队列中删除，并在 completion_counter 中增加完成任务的数量。
        """
        while not self.stop_event.is_set():
            try:
                task = self.queue.get_nowait()
            except queue.Empty:
                continue
            self.process_task(task)
            self.queue.task_done()
            with self.completion_counter.get_lock():
                self.completion_counter.value += 1

    def process_task(self, task):
        """
        处理一个任务。

        该方法将从队列中获取的任务交由 PressureUtil 的 sse_totally() 方法来处理，并在完成时打印完成的信息。
        在处理任务时，如果停止事件被设置，我们将不再处理该任务。
        """
        print(f"Queue {self.queue_id} processing {task}, start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        if not self.stop_event.is_set():
            pu = PressureUtil()
            user_id = int(pressure_config.get_value('pressure.num-users'))
            pu.sse_totally(self.queue_id, task, random.randint(1, user_id))
            print(f"Queue {self.queue_id} completed, end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

class TaskGenerator(threading.Thread):

    def __init__(self, queues, stop_event, mode, limit, completion_counters, interval=1):
        """
        任务生成器的构造函数。
        该函数将创建一个任务生成器，用于在多个队列中生成任务。可以选择在 count 模式下生成指定数量的任务，或者在 duration 模式下生成任务直到达到指定的时间限制。

        参数：
            queues (list[queue.Queue]): 任务队列的列表。
            stop_event (threading.Event): 一个用于指示何时停止生成任务的事件。
            mode (str): 任务生成的模式，可能的值为 'count' 或 'duration'。
            limit (int or float): 任务生成的限制，可能是一个数量限制（int）或一个时间限制（float，单位为秒）。
            completion_counters (dict[threading.Value]): 一个字典，用于跟踪每个队列中完成的任务数量。
            interval (int, optional): 任务生成的间隔时间（单位为秒），默认值为 1。

        属性：
            running (bool): 任务生成器当前是否正在运行的标志。
        """
        super().__init__()
        self.queues = queues
        self.stop_event = stop_event
        self.mode = mode
        self.limit = limit
        self.interval = interval
        self.running = True
        self.start_time = time.time()
        self.task_counters = {i: 0 for i in range(len(queues))}
        self.completion_counters = completion_counters

    def run(self):
        """
        任务生成器主循环。
        在这个循环中，我们会不断地在队列中生成任务，直到达到制定的限制为止。
        如果是 'count' 模式，我们会生成指定数量的任务，并在所有任务完成后停止。
        如果是 'duration' 模式，我们会生成任务直到达到指定的时间限制为止。
        """
        while not self.stop_event.is_set():
            if self.mode == 'count':
                if all(counter.value >= self.limit for counter in self.completion_counters):
                    self.stop_event.set()
                    break
                for i, q in enumerate(self.queues):
                    if self.task_counters[i] < self.limit:
                        q.put(f"Task_{i}_{self.task_counters[i]}")
                        self.task_counters[i] += 1
            elif self.mode == 'duration':
                if time.time() - self.start_time >= self.limit:
                    print(f"Duration limit of {self.limit} seconds reached. Stopping all tasks...")
                    self.stop_event.set()
                    break
                for i, q in enumerate(self.queues):
                    q.put(f"Task_{i}_{self.task_counters[i]}")
                    self.task_counters[i] += 1
            time.sleep(self.interval)


if __name__ == "__main__":
    # 加载测试数据
    loaded_data()
    
    # 线程停止信号
    stop_event = threading.Event()
    # 创建线程数组
    completion_counters = [multiprocessing.Value('i', 0) for _ in range(num_queues)]
    
    # 创建队列数组
    queues = [queue.Queue() for _ in range(num_queues)]
    # 创建任务处理线程
    for i, q in enumerate(queues):
        thread = TaskHandler(q, stop_event, i, completion_counters[i])
        threads.append(thread)
        thread.start()

    # 创建任务生成器线程
    generator_thread = TaskGenerator(queues, stop_event, mode, limit, completion_counters, interval)
    generator_thread.start()
    # 等待任务生成器线程完成
    generator_thread.join()

    # 等待所有任务处理线程完成
    for thread in threads:
        thread.join()

    # 打印最终统计信息
    print("Program completed!")
    for i, counter in enumerate(completion_counters):
        print(f"Queue {i} completed {counter.value} tasks")
    print(f"Total run time: {time.time() - generator_thread.start_time:.2f} seconds")
