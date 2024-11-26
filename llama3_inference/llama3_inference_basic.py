import asyncio
import torch
import subprocess
from huggingface_hub import login
from transformers import pipeline
import gradio as gr

from pypdf import PdfReader
from transformers import BitsAndBytesConfig
import time
from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

import os

# Log in to Hugging Face
login(token='Enter your token')

# CUDA and GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU setting

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/workspace/")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    device_map={"": 0}
    #cache_dir="/workspace/"
)

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 128000
model.config.eos_token_id = 128009

def format_message(message: str, history: list, memory_limit: int = 10) -> torch.Tensor:
    first_messages = [
        #{"role": "system", "content": ""},
        {"role": "user", "content": message},
    ]

    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return tokenizer.apply_chat_template(first_messages,
                                             add_generation_prompt=True,
                                             return_tensors="pt").to(device)

    formatted_message = [{"role": "system", "content": ""}]

    for user_msg, model_answer in history:
        formatted_message.append({"role": "user", "content": user_msg})
        formatted_message.append({"role": "assistant", "content": model_answer})
    formatted_message.append({"role": "user", "content": message})

    return tokenizer.apply_chat_template(formatted_message,
                                         add_generation_prompt=True,
                                         return_tensors="pt").to(device)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.stop_ids:
            return True
        return False

def generate_text(query, streamer):
    #asyncio.run(manage_layers_for_0th_layer())
    stop_criteria = StopOnTokens(stop_ids=[128009])

    output_ids = query
    past_key_values = None
    
    stopping_criteria_list = StoppingCriteriaList([stop_criteria])

    with torch.no_grad():
        for i in range(1000):  # Increase the number of iterations
            if past_key_values is None:
                outputs = model(input_ids=output_ids, use_cache=True)
            else:
                outputs = model(input_ids=output_ids[:, -1:], past_key_values=past_key_values, use_cache=True)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)

            #next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            #print(next_token)
            if i == 0:
            	streamer.put(next_token_id[0])
            streamer.put(next_token_id[0])

            past_key_values = outputs.past_key_values
            if stopping_criteria_list(input_ids=output_ids, scores=next_token_logits):
                torch.cuda.empty_cache()
                break
    streamer.end()

# Generate a response from the Llama model using custom token generation
def get_llama_response(message: str, history: list):
    if history is None:
        history = []
    query = format_message(message, history)  # output is tensor
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)

    def thread_func():
        try:
            generate_text(query, streamer)
        except Exception as e:
            print(f"Exception in thread: {e}")

    #t = Thread(target=generate_text, args=(query, streamer))
    t = Thread(target=thread_func)
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message

    t.join()

def upload_file(files):
    file_paths = [file.name for file in files]
    print(file_paths)
    return file_paths

def read_data(file):
    file_type = file[0].split(".")[1]
    if file_type == "txt":
        with open(file[0], 'r') as f:
            data = f.read()
        return data
    elif file_type == "pdf":
        reader = PdfReader(file[0])
        page = reader.pages[0]
        return page.extract_text()


CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chat { flex-grow: 1; overflow: auto;}
"""


with gr.Blocks(css= CSS, theme=gr.themes.Soft(text_size='lg')) as demo:
    with gr.Column(scale=500,):
        chat = gr.ChatInterface(get_llama_response, fill_height=True)
        torch.cuda.empty_cache()
    with gr.Column(scale=1):
        upload_button = gr.UploadButton("Click to Upload a File", file_types=["file"], file_count="multiple",size='sm', scale=1)
        upload_button.upload(upload_file, upload_button)
        upload_button.upload(read_data, upload_button, chat.textbox)
demo.launch(debug=True, share=True)

