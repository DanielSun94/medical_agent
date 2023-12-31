from typing import Dict, Optional
from torch.nn import Module
from config import logger
from accelerate import dispatch_model
import os


def set_visible_gpu(available_gpu_str):
    assert isinstance(available_gpu_str, str)
    gpus = available_gpu_str.strip().split(',')
    for item in gpus:
        logger.info('gpu {} is available'.format(item))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
    os.environ['CUDA_VISIBLE_DEVICES'] = available_gpu_str
    return gpus


def load_baichuan_model_on_gpus(model, gpu_id_list: list, device_map: Optional[Dict[str, int]] = None) -> Module:
    if device_map is not None:
        model = dispatch_model(model, device_map=device_map)
        return model

    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 40
    per_gpu_layers = (num_trans_layers+1) / len(gpu_id_list)

    gpu_id_dict = dict()
    for i, item in enumerate(gpu_id_list):
        gpu_id_dict[i] = item

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2
    device_map = {
        'model.embed_tokens': 0,
        'model.norm.weight': 1
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < len(gpu_id_list)
        device_map[f'model.layers.{i}'] = int(gpu_id_dict[gpu_target])
        used += 1
    model = dispatch_model(model, device_map=device_map)
    return model


def load_chatglm_model_on_gpus(model, gpu_id_list: list, device_map: Optional[Dict[str, int]] = None) -> Module:
    if device_map is not None:
        model = dispatch_model(model, device_map=device_map)
        return model

    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / len(gpu_id_list)

    gpu_id_dict = dict()
    for i, item in enumerate(gpu_id_list):
        gpu_id_dict[i] = item

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 1,
        'transformer.output_layer': 1,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < len(gpu_id_list)
        device_map[f'transformer.encoder.layers.{i}'] = int(gpu_id_dict[gpu_target])
        used += 1
    model = dispatch_model(model, device_map=device_map)
    return model


def load_tiger_model_on_gpus(model, gpu_id_list: list, device_map: Optional[Dict[str, int]] = None) -> Module:
    if device_map is not None:
        model = dispatch_model(model, device_map=device_map)
        return model

    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 80
    per_gpu_layers = 80 / len(gpu_id_list)

    gpu_id_dict = dict()
    for i, item in enumerate(gpu_id_list):
        gpu_id_dict[i] = item

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM2
    device_map = {
        'model.embed_tokens': 0,
        'model.norm.weight': 3,
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < len(gpu_id_list)
        device_map[f'model.layers.{i}'] = int(gpu_id_dict[gpu_target])
        used += 1
    model = dispatch_model(model, device_map=device_map)
    return model
