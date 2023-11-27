import os
import pickle
from config import logger, knowledge_target_folder, knowledge_source_folder
from config import args
import json, datetime
import csv
from itertools import islice
from utils import set_visible_gpu
import requests
import numpy as np
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline


logger.info('vector database sever start')
embedding_model_name = args['embedding_model_name']


def load_model(name, _):
    if name == 'corom-chinese-medical':
        model_id = "damo/nlp_corom_sentence-embedding_chinese-base-medical"
        pipeline_se = pipeline(Tasks.sentence_embedding, model=model_id)
    else:
        raise ValueError('')
    return pipeline_se


#
# def invoke_bge_reranker_large(model, json_post_list):
#     text_list = json_post_list.get('text')
#     embedding = model.encode(text_list)
#     arr_list = embedding.tolist()
#     json_str = json.dumps(arr_list)
#     time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     answer = {
#         "embedding": json_str,
#         "status": 200,
#         "time": time
#     }
#     log = "[" + time + "]  Embedding Generate Success"
#     print(log)
#     return answer
#

def invoke_text_embedding_model(model_name: str, text_list):
    if model_name == embedding_model_name:
        if model_name == 'corom-chinese-medical':
            response = invoke_corom(embedding_model, text_list)
        else:
            raise ValueError('')
    else:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = "[" + time + "]  Embedding Generate Failed"
        logger.info(log)
        response = []
    return response


def invoke_corom(model, text_list):
    embedding = model({'source_sentence': text_list})['text_embedding']
    arr_list = embedding.tolist()
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = "[" + time + "]  Embedding Generate Success"
    logger.info(log)
    return arr_list


# def query_embedding_server(data, model_name):
#     target_port = args['embedding_server_port']
#     url = "http://127.0.0.1:{}/text_embedding/query/{}".format(target_port, model_name)
#     headers = {'Content-Type': "application/json"}
#     payload = json.dumps({'text': data})
#     x = requests.post(url, headers=headers, data=payload)
#     response = x.json()
#
#     embedding = json.loads(response['embedding'])
#     return embedding


def initialize():
    logger.info("initialize start")
    file_list = os.listdir(knowledge_source_folder)
    target_folder = os.path.join(knowledge_target_folder, embedding_model_name)
    os.makedirs(target_folder, exist_ok=True)
    for file in file_list:
        base_name = ".".join(file.split('.')[0:-1])
        pkl_file = os.path.join(target_folder, base_name+'.pkl')
        if not os.path.exists(pkl_file):
            source_file = os.path.join(knowledge_source_folder, file)
            generate_embedding_db(embedding_model_name, source_file, pkl_file)
    logger.info("initialize success")


def generate_embedding_db(model_name, source, target, batch_size=2048):
    data_list = []
    embedding_list = []
    with open(source, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            key, content = line
            data_list.append([key, content])


    for i in range(len(data_list)//batch_size+1):
        if i < len(data_list) // batch_size:
            batch_data = data_list[i*batch_size: (i+1)*batch_size]
        else:
            batch_data = data_list[i*batch_size:]

        batch_key = [item[0] for item in batch_data]
        embedding = invoke_text_embedding_model(model_name, batch_key)
        for item in embedding:
            embedding_list.append(item)
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info('embedding generating: {}'.format(time))

    embedding_matrix = np.array(embedding_list)
    with open(target, 'wb') as f:
        pickle.dump({"vector": embedding_matrix, "knowledge": data_list}, f)


def load_vector():
    full_data = {}
    target_folder = os.path.join(knowledge_target_folder, embedding_model_name)
    for file in os.listdir(target_folder):
        base_name = ".".join(file.split('.')[0:-1])
        file_name = os.path.join(target_folder, file)
        full_data[base_name] = pickle.load(open(file_name, 'rb'))

    full_mat = []
    knowledge_map_dict = {}
    idx = 0
    for name in full_data:
        vectors = full_data[name]['vector']
        knowledge_list = full_data[name]['knowledge']
        for vec, knowledge in zip(vectors, knowledge_list):
            full_mat.append(vec)
            knowledge_map_dict[idx] = {'key': knowledge[0], "content": knowledge[1], 'doc': name}
            idx += 1

    vectors = np.stack(full_mat, axis=0)
    db = {'vectors': vectors, 'knowledge_map': knowledge_map_dict}
    return db


def get_top_n_idx(one_d_array, query_type, knowledge_list, n=1):
    target_list = []
    for i, item in enumerate(one_d_array):
        target_list.append([i, item])
    target_list = sorted(target_list, key=lambda x:x[1], reverse=True)

    target_idx = None
    for item in target_list:
        idx = item[0]
        doc = knowledge_list[idx]['doc']
        if query_type == "agent_knowledge" and doc == "agent_knowledge":
            target_idx = idx
            break
        # 之所以这么写是因为background_knowledge的doc name存在差异
        if query_type == 'background_knowledge' and doc != 'agent_knowledge':
            target_idx = idx
            break
    assert target_idx is not None
    return target_idx


gpu_list = set_visible_gpu(args['visible_gpu_ids'])
embedding_model = load_model(embedding_model_name, gpu_list)
# initialize process
logger.info('start initialize')
initialize()
logger.info('vector initialize success')
vector_db = load_vector()
logger.info('vector_db load success')


def query_background_knowledge(query, model_name, query_type):
    assert isinstance(query, str)
    assert query_type == 'agent_knowledge' or query_type == 'background_knowledge'

    embedding_mat, knowledge_list = vector_db['vectors'], vector_db['knowledge_map']
    query_embedding = np.squeeze(np.array(invoke_text_embedding_model(model_name, [query])))
    embedding_norm, query_norm = np.linalg.norm(embedding_mat, axis=1), np.linalg.norm(query_embedding)
    similarity_mat = np.matmul(embedding_mat, query_embedding) / (embedding_norm * query_norm)

    target_idx = get_top_n_idx(similarity_mat, query_type, knowledge_list)
    prompt = knowledge_list[target_idx]['content']
    return prompt
