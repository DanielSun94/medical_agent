import os
import logging
import argparse


huggingface_cache_folder = '/home/disk/sunzhoujian/hugginface'
knowledge_source_folder = os.path.abspath('./resource/knowledge/source')
knowledge_target_folder = os.path.abspath('./resource/knowledge/target/')
modelscope_cache_folder = '/home/sunzhoujian/modelscope'


log_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'resource', 'log')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_file_name = os.path.join(log_folder, '{}.txt'.format('fetch_guideline_log'))
format_ = "%(asctime)s %(process)d %(module)s %(lineno)d %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(format_)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


parser = argparse.ArgumentParser()
parser.add_argument('--visible_gpu_ids', help='', default='0', type=str)
parser.add_argument('--llm_name', help='', type=str)
parser.add_argument('--embedding_model_name', help='', type=str)
parser.add_argument('--port', help='', type=str)
args = vars(parser.parse_args())
