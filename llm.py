import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from utils import load_chatglm_model_on_gpus
from config import huggingface_cache_folder, args, logger
import requests
import json
import datetime
from utils import set_visible_gpu
import yaml


def load_model(name, visible_gpu_list):
    if name == 'chatglm2-6b':
        backbone = "THUDM/chatglm2-6b"
        tokenizer = AutoTokenizer.from_pretrained(
            backbone, trust_remote_code=True, cache_dir=huggingface_cache_folder)
        model = AutoModel.from_pretrained(
            backbone, trust_remote_code=True, cache_dir=huggingface_cache_folder)
        if len(visible_gpu_list) > 1:
            model = load_chatglm_model_on_gpus(model, visible_gpu_list)
        else:
            model = model.cuda()
    elif name == 'chatglm3-6b':
        backbone = "THUDM/chatglm3-6b"
        tokenizer = AutoTokenizer.from_pretrained(
            backbone, trust_remote_code=True, cache_dir=huggingface_cache_folder)
        model = AutoModel.from_pretrained(
            backbone, trust_remote_code=True, cache_dir=huggingface_cache_folder)
        if len(visible_gpu_list) > 1:
            model = load_chatglm_model_on_gpus(model, visible_gpu_list)
        else:
            model = model.cuda()
    elif name == 'tiger-70b':
        backbone = "TigerResearch/tigerbot-70b-chat-v3"
        tokenizer = AutoTokenizer.from_pretrained(
            backbone, trust_remote_code=True, cache_dir=huggingface_cache_folder)
        model = AutoModelForCausalLM.from_pretrained(
            backbone, trust_remote_code=True, device_map='auto', cache_dir=huggingface_cache_folder).half()
        # if len(visible_gpu_list) > 1:
        #     llm_model = load_chatglm_model_on_gpus(llm_model, visible_gpu_list)
        # else:
        #     llm_model = llm_model.cuda()
    elif name == 'baichuan2-13b-chat':
        tokenizer = AutoTokenizer.from_pretrained(
            "baichuan-inc/Baichuan2-13B-Chat", use_fast=False, trust_remote_code=True,
            cache_dir=huggingface_cache_folder)
        model = AutoModelForCausalLM.from_pretrained(
            "baichuan-inc/Baichuan2-13B-Chat", device_map="auto", torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=huggingface_cache_folder)
        model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat")
    # elif name == 'bge-reranker-large':
    #     model = SentenceTransformer('BAAI/bge-reranker-large', cache_folder=huggingface_cache_folder).to('cuda:1')
    #     available_model_dict[name] = model
    # elif name == 'corom-chinese-medical':
    #     model_id = "damo/nlp_corom_sentence-embedding_chinese-base-medical"
    #     pipeline_se = pipeline(Tasks.sentence_embedding, model=model_id)
    #     available_model_dict[name] = pipeline_se
    else:
        raise ValueError('')
    return tokenizer, model


def invoke_llm_model(model_name: str, user_utt, history):
    if model_name != llm_name:
        if model_name == 'ernie_bot':
            response = invoke_ernie_bot(user_utt, history)
        else:
            time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = "Error", ["Error", "Error"]
            logger.info('error at {}'.format(time))
    elif model_name == 'tiger-70b':
        response, history = invoke_tiger_70b(llm_model, llm_tokenizer, user_utt, history)
    elif model_name == 'chatglm2-6b':
        response, history = invoke_chatglm2_6b(llm_model, llm_tokenizer, user_utt, history)
    elif model_name == 'chatglm3-6b':
        response, history = invoke_chatglm3_6b(llm_model, llm_tokenizer, user_utt, history)
    elif model_name == 'baichuan2-13b-chat':
        response, history = invoke_baichuan2_13b(llm_model, llm_tokenizer, user_utt, history)
    else:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = "Error", ["Error", "Error"]
        logger.info('error at {}'.format(time))
    return response, history


def invoke_ernie_bot(user_utt, history):
    url = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions'
    with open("resource/ernie_bot/ernie_bot_access_token.yaml", 'r', encoding='utf-8-sig') as f:
        resource = yaml.load(f, yaml.FullLoader)

    access_token = resource['access_token']
    headers = {'Content-Type': "application/json"}
    params = {'access_token': access_token}

    assert len(history) % 2 == 0
    message = []
    for i, utterance in enumerate(history):
        if i % 2 == 0:
            message.append({"role": "user", "content": utterance})
        else:
            message.append({"role": "assistant", "content": utterance})
    message.append({"role": "user", "content": user_utt})

    payload = json.dumps({"messages": message})
    x = requests.post(url, headers=headers, params=params, data=payload)
    response = x.json()

    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if x.status_code == 200:
        result = response['result']
        message.append({"role": "assistant", "content": result})
        history = [item['content'] for item in message]

        log = "[" + time + "] " + '", user input:"' + user_utt + '", response:"' + repr(response) + '"'
        logger.info(log)
        return response, history
    else:
        return "Error", ["Error", "Error"]


def invoke_chatglm3_6b(model, tokenizer, user_utt, history):

    assert len(history) % 2 == 0
    message = []
    for i in range(len(history) // 2):
        message.append({'role': "user", "content": history[i*2]})
        message.append({'role': "assistant", "metadata": "", "content": history[(i*2)+1]})

    response, history = model.chat(tokenizer, user_utt, history=message)

    history_list = []
    for item in history:
        history_list.append(item['content'])

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    log = "[" + time + "] " + '", prompt:"' + user_utt + '", response:"' + repr(response) + '"'
    logger.info(log)
    return response, history


def invoke_chatglm2_6b(model, tokenizer, user_utt, history):
    assert len(history) % 2 == 0
    message = []
    for i in range(len(history) // 2):
        message.append([history[i * 2], history[i * 2 + 1]])

    response, history = model.chat(tokenizer, user_utt, history=message)

    history_list = []
    for item in history:
        for element in item:
            history_list.append(element)

    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = "[" + time + "] " + '", prompt:"' + user_utt + '", response:"' + repr(response) + '"'
    log(log)
    return response, history


def invoke_tiger_70b(model, tokenizer, user_utt, history):

    assert len(history) % 2 == 0
    message = []
    for i, utterance in enumerate(history):
        if i % 2 == 0:
            message.append({"role": "user", "content": utterance})
        else:
            message.append({"role": "assistant", "content": utterance})

    message.append({"role": "user", "content": user_utt})
    response = model.chat(tokenizer, message)
    message.append({"role": "assistant", "content": response})
    history = [item['content'] for item in message]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    log = "[" + time + "] " + '", user input:"' + user_utt + '", response:"' + repr(response) + '"'
    logger.info(log)
    return response, history


def invoke_baichuan2_13b(model, tokenizer, user_utt, history):
    assert len(history) % 2 == 0
    message = []
    for i, utterance in enumerate(history):
        if i % 2 == 0:
            message.append({"role": "user", "content": utterance})
        else:
            message.append({"role": "assistant", "content": utterance})

    message.append({"role": "user", "content": user_utt})
    response = model.chat(tokenizer, message)
    message.append({"role": "assistant", "content": response})
    history = [item['content'] for item in message]

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    log = "[" + time + "] " + '", user input:"' + user_utt + '", response:"' + repr(response) + '"'
    logger.info(log)
    return response, history



gpu_list = set_visible_gpu(args['visible_gpu_ids'])
llm_name = args['llm_name']
llm_tokenizer, llm_model = load_model(llm_name, gpu_list)
