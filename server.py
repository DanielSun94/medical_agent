from fastapi import FastAPI, Request
import uvicorn, json
from llm import invoke_llm_model
from vec_db import query_background_knowledge
from config import logger, args
from fastapi.middleware.cors import CORSMiddleware


logger.info('llm sever start')
llm_name = args['llm_name']
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或者是你的前端应用的域名列表
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)
valid_model = args

"""
TBD Fast API作为后端使用
需要有保存用户数据，包括但不限于用户画像
需要进行登录管理
"""


@app.post("/rag_agent/{llm_model_name}/{embedding_model_name}")
async def rag_agent_predict(llm_model_name: str, embedding_model_name: str, request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    history = json_post_list.get('history')
    user_utterance = json_post_list.get('input')

    assert isinstance(user_utterance, str)
    assert all(isinstance(item, str) for item in history)

    background_knowledge = ''
    if len(history) == 0:
        background_knowledge = query_background_knowledge(user_utterance, embedding_model_name, 'agent_knowledge')
        textbox = '请你扮演一名医生，请严格根据以下决策过程，从问题1开始向我提问，了解我的情况，最终得出结论。每轮对话允许只问一个问题，' \
                  '不允许问决策过程中没有出现的问题。\n{}\n'.format(background_knowledge)
        user_input = textbox
    else:
        user_input = user_utterance

    response, history = invoke_llm_model(llm_model_name, user_input, history)

    if len(history) == 2:
        history[0] = user_utterance

    assert isinstance(background_knowledge, str)
    assert isinstance(response, str)
    assert all(isinstance(item, str) for item in history)
    return {
        "response": response,
        'history': history,
        'background_knowledge': background_knowledge
    }


@app.post("/rag_qa/{llm_model_name}/{embedding_model_name}")
async def rag_qa_predict(llm_model_name: str, embedding_model_name: str, request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    history = json_post_list.get('history')
    user_utterance = json_post_list.get('input')

    assert isinstance(user_utterance, str)
    assert all(isinstance(item, str) for item in history)

    # context_list = (history + [user_utterance])
    # context_list.reverse()
    # query = '\n'.join(context_list)
    background_knowledge = query_background_knowledge(user_utterance, embedding_model_name, 'background_knowledge')
    background_knowledge = '请根据以下背景知识作答: {}'.format(background_knowledge)
    user_utterance_input = background_knowledge + '\n' + user_utterance

    response, history = invoke_llm_model(llm_model_name, user_utterance_input, history)
    # 在输出里把prompt屏蔽掉
    history[-2] = user_utterance
    assert isinstance(background_knowledge, str)
    assert isinstance(response, str)
    assert all(isinstance(item, str) for item in history)
    return {
        "response": response,
        'history': history,
        'background_knowledge': background_knowledge
    }


@app.post("/vanilla_chat/{llm_model_name}")
async def vanilla_predict(llm_model_name: str, request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    history = json_post_list.get('history')
    user_utterance = json_post_list.get('input')

    response, history = invoke_llm_model(llm_model_name, user_utterance, history)
    return {
        "response": response,
        'history': history,
        'background_knowledge': "none"
    }



if __name__ == '__main__':
    port = int(args['port'])
    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)
    logger.info('app running')