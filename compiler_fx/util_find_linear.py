import torch
from transformers import AutoModel
import sys
import os

def find_linear_modules(model):
    linear_modules = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_modules.append((name, module))

    return linear_modules

def find_linear_modules2(model):
    linear_modules = []

    def recursive_search(prefix, module):
        for name, sub_module in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(sub_module, torch.nn.Linear):
            #if isinstance(sub_module, torch.nn.Linear) or isinstance(sub_module, torch.nn.Conv1d) or hasattr(sub_module, "weight"):
                linear_modules.append((full_name, sub_module))
            else:
                recursive_search(full_name, sub_module)

    recursive_search("", model)

    return linear_modules

if __name__ == "__main__":
    #print(f"len(sys.argv) --> {len(sys.argv)}")
    #print(f"sys.argv[0] --> {sys.argv[0]}")


    #model_name = "bert-base-uncased" 
    #model_name = "openai/whisper-base" 
    model_name = "facebook/opt-350m" 

    model = AutoModel.from_pretrained(model_name)

    #print(f"model: {model}")

    linear_layers = find_linear_modules(model)
    #linear_layers = find_linear_modules2(model)

    if linear_layers:
        for name, module in linear_layers:
            print(f"- {name}: {module}")
        print(f">> found: {len(linear_layers)} nn.Linear")
    else:
        print("Not find nn.Linear")

