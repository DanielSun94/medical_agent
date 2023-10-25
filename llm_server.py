from fastapi import FastAPI, Request
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import uvicorn, json, datetime
from utils import load_chatglm_model_on_gpus
from sentence_transformers import SentenceTransformer
import argparse
import requests
import yaml
import os
from config import logger

logger.info('llm sever start')


app = FastAPI()
parser = argparse.ArgumentParser()
parser.add_argument('--visible_gpu_ids', help='', default='0', type=str)
parser.add_argument('--local_model_name', help='', type=str)
parser.add_argument('--port', help='', type=str)
args = vars(parser.parse_args())

cache_folder = '/home/disk/sunzhoujian/hugginface'


def load_model(model_name_list, visible_gpu_list):
    available_model_dict = dict()
    for name in model_name_list:
        if name == 'chatglm2-6b':
            backbone = "THUDM/chatglm2-6b"
            tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True, cache_dir=cache_folder)
            llm_model = AutoModel.from_pretrained(backbone, trust_remote_code=True, cache_dir=cache_folder)
            if len(visible_gpu_list) > 1:
                llm_model = load_chatglm_model_on_gpus(llm_model, visible_gpu_list)
            else:
                llm_model = llm_model.cuda()
            available_model_dict[name] = tokenizer, llm_model
        elif name == 'baichuan2-13b-chat':
            tokenizer = AutoTokenizer.from_pretrained(
                "baichuan-inc/Baichuan2-13B-Chat", use_fast=False, trust_remote_code=True, cache_dir=cache_folder)
            llm_model = AutoModelForCausalLM.from_pretrained(
                "baichuan-inc/Baichuan2-13B-Chat", device_map="auto", torch_dtype=torch.bfloat16,
                trust_remote_code=True, cache_dir=cache_folder)
            llm_model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat")
            available_model_dict[name] = tokenizer, llm_model
        elif name == 'bge-reranker-large':
            model = SentenceTransformer('BAAI/bge-reranker-large', cache_folder=cache_folder).to('cuda:1')
            available_model_dict[name] = model
        else:
            raise ValueError('')
    return available_model_dict


def set_visible_gpu(available_gpu_str):
    assert isinstance(available_gpu_str, str)
    gpus = available_gpu_str.strip().split(',')
    for item in gpus:
        print('gpu {} is available'.format(item))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
    os.environ['CUDA_VISIBLE_DEVICES'] = available_gpu_str
    return gpus


gpu_list = set_visible_gpu(args['visible_gpu_ids'])
local_models = args['local_model_name'].strip().split(';')
local_model_dict = load_model(local_models, gpu_list)


def invoke_bge_reranker_large(model, json_post_list):
    text_list = json_post_list.get('text')
    embedding = model.encode(text_list)
    arr_list = embedding.tolist()
    json_str = json.dumps(arr_list)
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "embedding": json_str,
        "status": 200,
        "time": time
    }
    log = "[" + time + "]  Embedding Generate Success"
    print(log)
    return answer


def invoke_ernie_bot(json_post_list):
    url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions'
    with open("resource/ernie_bot/ernie_bot_access_token.yaml", 'r', encoding='utf-8-sig') as f:
        resource = yaml.load(f, yaml.FullLoader)

    access_token = resource['access_token']
    headers = {'Content-Type': "application/json"}
    params = {'access_token': access_token}

    user_input_ = json_post_list.get('input')
    history = json_post_list.get('history')

    assert len(history) % 2 == 0
    message = []
    for i, utterance in enumerate(history):
        if i % 2 == 0:
            message.append({"role": "user", "content": utterance})
        else:
            message.append({"role": "assistant", "content": utterance})
    message.append({"role": "user", "content": user_input_})

    payload = json.dumps({"messages": message})
    x = requests.post(url, headers=headers, params=params, data=payload)
    response = x.json()

    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if x.status_code == 200:
        result = response['result']
        message.append({"role": "assistant", "content": result})
        history = [item['content'] for item in message]
        answer = {
            "response": result,
            "history": history,
            "status": 200,
            "time": time
        }
        log = "[" + time + "] " + '", user input:"' + user_input_ + '", response:"' + repr(response) + '"'
        print(log)
        return answer
    else:
        return {
            "response": 'error',
            "history": [],
            "status": 200,
            "time": time
        }


def invoke_chatglm2_6b(models, json_post_list):
    tokenizer, model = models

    user_input_ = json_post_list.get('input')
    history = json_post_list.get('history')
    # max_length = json_post_list.get('max_length')
    # top_p = json_post_list.get('top_p')
    # temperature = json_post_list.get('temperature')

    # max_length = max_length if max_length else 2048
    # top_p = top_p if top_p else 0.7

    assert len(history) % 2 == 0
    message = []
    for i in range(len(history) // 2):
        message.append([history[i*2], history[i*2+1]])

    response, history = model.chat(tokenizer, user_input_, history=message)
                                   # max_length=max_length,
                                   # top_p=top_p,
                                   # temperature=temperature if temperature else 0.95)

    history.append(user_input_)
    history.append(response)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + user_input_ + '", response:"' + repr(response) + '"'
    print(log)
    return answer


def invoke_baichuan2_13b(models, json_post_list):
    tokenizer, model = models

    user_input = json_post_list.get('input')
    history = json_post_list.get('history')

    assert len(history) % 2 == 0
    message = []
    for i, utterance in enumerate(history):
        if i % 2 == 0:
            message.append({"role": "user", "content": utterance})
        else:
            message.append({"role": "assistant", "content": utterance})

    message.append({"role": "user", "content": user_input})
    response = model.chat(tokenizer, message)
    message.append({"role": "assistant", "content": response})
    history = [item['content'] for item in message]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", user input:"' + user_input + '", response:"' + repr(response) + '"'
    print(log)
    return answer


@app.post("/text_embedding/{model_name}")
async def invoke_model(model_name: str, request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    if model_name in local_model_dict:
        assert model_name == 'bge-reranker-large'
        model = local_model_dict[model_name]
        response = invoke_bge_reranker_large(model, json_post_list)
    else:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = {
            "response": '',
            "status": 404,
            "time": time
        }
    return response


@app.post("/chat/{model_name}")
async def invoke_model(model_name: str, request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    if model_name not in local_model_dict:
        if model_name == 'ernie_bot':
            response = invoke_ernie_bot(json_post_list)
        else:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = {
                "response": 'target model unavailable, available local model: {}'.format(args['model_name']),
                "status": 200,
                "time": time
            }
    elif model_name == 'chatglm2-6b':
        models = local_model_dict[model_name]
        response = invoke_chatglm2_6b(models, json_post_list)
    elif model_name == 'baichuan2-13b-chat':
        models = local_model_dict[model_name]
        response = invoke_baichuan2_13b(models, json_post_list)
    else:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = {
            "response": '',
            "status": 404,
            "time": time
        }
    return response


if __name__ == '__main__':
    port = int(args['port'])
    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)