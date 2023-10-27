import numpy as np
import gradio as gr
import argparse
import mdtex2html
import requests
import json
from config import logger

logger.info('web demo')


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', help='', type=str)
parser.add_argument('--llm_port', help='', default='8000', type=str)
parser.add_argument('--text_embedding_port', help='', default='8001', type=str)
parser.add_argument('--port', help='', default='7860', type=str)
args = vars(parser.parse_args())


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text



def find_top_n_prompt(similarity, content, n):
    similarity_list = []
    for i in range(len(similarity)):
        similarity_list.append((i, similarity[i]))
    similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)
    prompt = '背景知识：'
    for i in range(n):
        prompt = prompt + content[similarity_list[i][0]] + '。'
    return prompt


def query_knowledge(user_input_):
    url = "http://127.0.0.1:8001/text_embedding/"
    headers = {'Content-Type': 'application/json'}
    data = {'model_name': args['model_name'], 'query': user_input_, "top_n": "15"}
    payload = json.dumps(data)

    x = requests.post(url, headers=headers, data=payload)
    response = x.json()
    back_know = response['background_knowledge']
    return back_know



def rag_predict(current_model, chatbot, textbox, user_input, messages):
    if len(messages) == 0:
        back_know = query_knowledge(user_input)
        textbox = '请根据以下背景知识作答: {}'.format(back_know)
        messages = [textbox, "好的"]
    chatbot.append((parse_text(user_input), ""))
    response, history = invoke_llm_model(current_model, user_input, messages)
    chatbot[-1] = (parse_text(user_input), parse_text(response))
    return chatbot, textbox, history


def vanilla_predict(model_name_, chatbot_, _, user_input_, message_list):
    chatbot_.append((parse_text(user_input_), ""))
    response, history = invoke_llm_model(model_name_, user_input_, message_list)
    chatbot_[-1] = (parse_text(user_input_), parse_text(response))
    return chatbot_, _, history


def predict(query_type, current_model, chatbot, textbox, user_input, messages):
    query_type = query_type.lower()
    current_model = current_model.lower()
    if query_type == 'vanilla':
        return vanilla_predict(current_model, chatbot, textbox, user_input, messages)
    else:
        return rag_predict(current_model, chatbot, textbox, user_input, messages)


def invoke_llm_model(model_name_, user_input_, message_list):
    port = int(args['llm_port'])

    data = {
        'input': user_input_,
        'history': message_list,
    }
    data_json = json.dumps(data)
    url = 'http://localhost:{}/chat/{}'.format(port, model_name_)
    response = requests.post(url, data=data_json, headers={'Content-Type': 'application/json'})

    json_post_raw = response.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    response = json_post_list.get('response')
    history = json_post_list.get('history')
    return response, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


def get_value(value):
    return value


with gr.Blocks() as demo:
    current_model = gr.State("ERNIE BOT")
    query_type = gr.State("Vanilla")
    gr.HTML("""<h1 align="center">LLM Test</h1>""")

    textbox = gr.Textbox(label="RAG Prompt", info="RAG Prompt", lines=3, value="None", interactive=False)
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                model_select_box = gr.Dropdown(["ERNIE_Bot", "ChatGLM2-6B", "Baichuan2-13B-Chat"], label="Models")
                query_select_box = gr.Dropdown(["Vanilla", "RAG"], label="Query Type")
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
            with gr.Column(min_width=32, scale=1):
                submitBtn1 = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            # max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            # top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            # temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    messages = gr.State([])
    model_select_box.change(get_value, [model_select_box], [current_model])
    query_select_box.change(get_value, [query_select_box], [query_type])

    submitBtn1.click(predict, [query_type, current_model, chatbot, textbox, user_input, messages],
                     [chatbot, textbox, messages], show_progress=True)
    submitBtn1.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[chatbot, messages], show_progress=True)



demo.queue().launch(server_name='0.0.0.0', server_port=int(args['port']), share=False, inbrowser=True)