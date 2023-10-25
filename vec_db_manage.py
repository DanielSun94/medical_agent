import os
import pickle

from config import logger
from fastapi import FastAPI, Request
import argparse
import uvicorn, json, datetime
import csv
from itertools import islice
import requests
import numpy as np

logger.info('vector database sever start')


app = FastAPI()
parser = argparse.ArgumentParser()
parser.add_argument('--port', help='', type=str)
parser.add_argument('--embedding_server_port', help='', type=str)
parser.add_argument('--embedding_model', help='', type=str)
args = vars(parser.parse_args())

knowledge_folder = os.path.abspath('./resource/knowledge/source')
save_folder = os.path.abspath('./resource/knowledge/target/')


def query_embedding_server(data, model_name):
    target_port = args['embedding_server_port']
    url = "http://127.0.0.1:{}/text_embedding/{}".format(target_port, model_name)
    headers = {'Content-Type': "application/json"}
    payload = json.dumps({'text': data})
    x = requests.post(url, headers=headers, data=payload)
    response = x.json()

    embedding = json.loads(response['embedding'])
    return embedding


def initialize():
    model_name = args['embedding_model']
    file_list = os.listdir(knowledge_folder)
    target_folder = os.path.join(save_folder, model_name)
    os.makedirs(target_folder, exist_ok=True)
    for file in file_list:
        base_name = ".".join(file.split('.')[0:-1])
        pkl_file = os.path.join(target_folder, base_name+'.pkl')
        if not os.path.exists(pkl_file):
            source_file = os.path.join(knowledge_folder, file)
            generate_embedding_db(model_name, source_file, pkl_file)
    logger.info("initialize success")


def generate_embedding_db(model_name, source, target, batch_size=2048):
    data_list = []
    embedding_list = []
    with open(source, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        idx = 0
        for line in csv_reader:
            if idx == 0:
                idx += 1
                continue
            key, content = line
            data_list.append([key, content])


    for i in range(len(data_list)//batch_size+1):
        if i < len(data_list) // batch_size:
            batch_data = data_list[i*batch_size: (i+1)*batch_size]
        else:
            batch_data = data_list[i*batch_size:]

        batch_key = [item[0] for item in batch_data]
        embedding = query_embedding_server(batch_key, model_name)
        for item in embedding:
            embedding_list.append(item)
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info('embedding generating: {}'.format(time))

    embedding_matrix = np.array(embedding_list)
    with open(target, 'wb') as f:
        pickle.dump({"vector": embedding_matrix, "knowledge": data_list}, f)


def load_vector():
    full_data = {}
    model_name = args['embedding_model']
    target_folder = os.path.join(save_folder, model_name)
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

def get_top_n_idx(one_d_array, top_n):
    target_list = []
    for i, item in enumerate(one_d_array):
        target_list.append([i, item])
    target_list = sorted(target_list, key=lambda x:x[1], reverse=True)
    return target_list[:top_n]

# initialize process
initialize()
vector_db = load_vector()


@app.post("/text_embedding/")
async def get_embedding(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    available_models = os.listdir(save_folder)
    model_name = json_post_list.get("model_name")
    query = json_post_list.get("query")
    top_n = int(json_post_list.get("top_n"))
    assert model_name in available_models
    assert isinstance(query, str)

    embedding_mat, knowledge_list = vector_db['vectors'], vector_db['knowledge_map']
    query_embedding = np.array(query_embedding_server(query, model_name))
    similarity_mat = np.matmul(embedding_mat, query_embedding)


    top_n_indices = get_top_n_idx(similarity_mat, top_n)
    prompt_list = [knowledge_list[idx[0]]['content'] for idx in top_n_indices]
    prompt = "; ".join(prompt_list)
    return {'background_knowledge': prompt}


if __name__ == '__main__':
    port = int(args['port'])
    uvicorn.run(app, host='0.0.0.0', port=port, workers=1)