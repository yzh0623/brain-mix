"""
Copyright (c) 2024 by paohe information technology Co., Ltd. All right reserved.
FilePath: /brain-mix/run.py
Author: yuanzhenhui
Date: 2024-04-04 23:01:32
LastEditTime: 2025-01-14 22:47:13
"""

from flask import Flask, Response, request, make_response, render_template
from flask_cors import CORS

import json

import os
project_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(project_dir, 'utils'))
sys.path.append(os.path.join(project_dir, 'utils','llms'))
sys.path.append(os.path.join(project_dir, 'rag'))
import common_util as CommonUtil
from yaml_util import YamlConfig
from logging_util import LoggingUtil
logger = LoggingUtil(os.path.basename(__file__).replace(".py", ""))

from session_context import SessionContext
from knowledge_search import KnowledgeSearch
from cuda_util import CudaMultiProcessor

web_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'web_cnf.yml'))
http_host = web_config.get_value('http.host')
http_port = web_config.get_value('http.port')
http_content_type = web_config.get_value('http.content_type')

base_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'llms', 'base_cnf.yml'))
base_prompt = base_config.get_value('llm.system.prompt')

app = Flask(__name__)
CORS(app)

@app.before_request
def initialize_app():
    try:
        if not app.config.get('initialized'):
            
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            app.session_instance = SessionContext()
            app.knowledge_instance = KnowledgeSearch()
            app.cuda_multi_processor = CudaMultiProcessor()
            
            pytorch_config = YamlConfig(os.path.join(project_dir, 'resources', 'config', 'llms', 'pytorch_cnf.yml'))
            multi_core = int(pytorch_config.get_value('pytorch.multi_core'))
            app.switch_thrid = pytorch_config.get_value('pytorch.switch_thrid')
            
            logger.info("begin to load process models...")
            for _ in range(multi_core):
                message = {"role": "user", "content": "hello!"}
                queue_id = app.cuda_multi_processor.start_generation(message,False)
                app.cuda_multi_processor.get_results_by_queueid(queue_id)
            
            app.config['initialized'] = True
    except Exception as e:
        logger.error(f"instance initialization failed: {e}")
        raise

@app.route('/')
def chat_page():
    return render_template('chat.html')

@app.route('/api/load_session', methods=['POST'])
def load_session():
    user_id = request.json.get('user_id')
    pick_latest = request.json.get('pick_latest')
    sessions = app.session_instance.get_user_sessions(user_id, pick_latest)
    response = make_response(json.dumps(sessions))
    response.headers['Content-Type'] = http_content_type
    return response
  
@app.route('/api/load_context', methods=['POST'])
def load_context():
    session_id = request.json.get('session_id')
    current_page = request.json.get('current_page')
    _,chat_context = app.session_instance.get_user_context(session_id,int(current_page), [])
    response_data = json.dumps(chat_context)
    response = make_response(response_data)
    response.headers['Content-Type'] = http_content_type
    return response

@app.route('/api/delete_session_by_id', methods=['POST'])
def delete_session_by_id():
    session_id = request.json.get('session_id')
    app.session_instance.delete_user_sessions(session_id)
    response = make_response(json.dumps({'message': 'Session deleted successfully'}))
    response.headers['Content-Type'] = http_content_type
    return response

@app.route('/api/edit_session_title_by_id', methods=['POST'])
def edit_session_title_by_id():
    session_id = request.json.get('session_id')
    title = request.json.get('title')
    result = app.session_instance.update_session_title(session_id,title)
    if result == 1:
        response = make_response(json.dumps({'retcode':1,'message': 'Session edit successfully'}))
    else:
        response = make_response(json.dumps({'retcode':0,'message': 'Session edit falused'}))
    response.headers['Content-Type'] = http_content_type
    return response

@app.route('/api/delete_context_by_id', methods=['POST'])
def delete_context_by_id():
    session_id = request.json.get('session_id')
    context_id = request.json.get('context_id')
    result = app.session_instance.delete_user_context_from_contextid(session_id,context_id)
    if result > 0:
        response = make_response(json.dumps({'retcode':1,'message': 'Question delete successfully'}))
    else:
        response = make_response(json.dumps({'retcode':0,'message': 'Question delete falused'}))
    response.headers['Content-Type'] = http_content_type
    return response

@app.route('/api/chat_generation', methods=['POST'])
def chat_generation():
    first_flag = False
    user_id = request.json.get('user_id', '')
    session_id = request.json.get('session_id', '')
    messages = request.json.get('messages', [])

    if session_id == '':
        first_flag = True
        session_id = app.session_instance.save_user_session(user_id, messages[0].get('content', '')[:20])
    context_id,all_context = app.session_instance.get_user_context(session_id, -1, messages)
    need_summary = app.knowledge_instance.find_summary_search(all_context)
    response_text = []
    save_response = []
    def generate():
        callback_context = {}
        skip_flag = True
        queue_id = app.cuda_multi_processor.start_generation(need_summary,app.switch_thrid)
        for chunk in app.cuda_multi_processor.get_results_by_queueid(queue_id):
            chunk_text = CommonUtil.clean_markdown(chunk["text"])
            response_text.append(chunk_text)
            if '[' in chunk_text:
                skip_flag = False
            if skip_flag:
                save_response.append(chunk_text)
                yield f"""data: {
                    json.dumps({
                            'text': chunk_text,
                            'token_count': chunk['token_count'], 
                            'total_token_count': chunk['total_token_count'], 
                            'token_rate': chunk['token_rate']
                            }, ensure_ascii=False)
                        }\n\n"""
            
        yield f"""data: {
            json.dumps({
                'context_id':context_id
                }, ensure_ascii=False)
            }\n\n"""
        
        latest_response = ''.join(response_text)
        latest_response_arr = latest_response.split('[')
        if len(latest_response_arr) > 1:
            guess_context = '['+latest_response_arr[1]
            callback_context = {"guess_context": guess_context}
        
        yield f"data: {json.dumps({'callback_context':callback_context}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'first_flag': first_flag, 'session_id':session_id, 'title': messages[0].get('content', '')[:20]}, ensure_ascii=False)}\n\n"
        yield f"data: [DONE]\n\n"
        
        app.session_instance.save_user_context(session_id, [{"role": "assistant", "content": ''.join(save_response)}])
    return Response(generate(), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(host=http_host, port=http_port, debug=False)