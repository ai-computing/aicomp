#! /usr/bin/env python
import numpy as np
import pandas as pd
import os, sys, argparse

# def get_args():
#     filename = "428087"
#     layer_number = 48
#     num_mb = 2
#     return filename, layer_number, num_mb

def get_args():
    filename = "tp_output"
    num_mb = 1
    return filename, num_mb

def get_logpath(filename):
    j = filename
    # print(j)
    #log_path = os.path.join(os.getcwd(),"log2",j)
    parent_dir = os.path.dirname(os.getcwd())
    log_path = os.path.join(parent_dir, "log2", j) 
    # print(log_path)
    # print(os.listdir(log_path))
    log_list = sorted([ f for f in os.listdir(log_path) if f.endswith('.out')])
    # print(log_list)
    log_path = os.path.join(log_path, log_list[0])
    # print(log_path)
    return log_path

def main():

    j, num_mb = get_args()
    sign_array = []
    # profile_name = ""
    path = get_logpath(j)
    times_array = []
    
    sign_em = "Embedding layer:"
    sign_layer = "layer time:"
    sign_post = "post_process:"
    sign_array.append(sign_em)
    sign_array.append(sign_layer)
    sign_array.append(sign_post)
    embedding_times, layer_times, post_process = get_layer_time(path,sign_array,num_mb)
    
    df_layer = pd.DataFrame(layer_times, columns=["time"], dtype=float)
    df_layer["time"] = pd.to_numeric(df_layer["time"])
    
    df_emb_layer = pd.DataFrame(embedding_times, columns=["time"], dtype=float)
    df_emb_layer["time"] = pd.to_numeric(df_emb_layer["time"])
    
    df_post_process = pd.DataFrame(post_process)
    
    print(f"layer time len: {df_layer.size}")
    # print(f"layer time avg: {df_layer.mean()}")
    # print(df_layer.describe())
    # print(df_layer.loc[df_layer["time"] < 30.0].describe())
    # print(df_layer.loc[df_layer["time"] < 7.0].describe())
    print(df_layer.describe())
    
    # df_layer.to_csv("A10_layer_profile_TP2.csv")
    
    print(f"emb layer time len: {df_emb_layer.size}")
    # print(f"emb layer time avg: {df_emb_layer.mean()}")
    # print(df_emb_layer.describe())
    # print(df_emb_layer.loc[df_emb_layer["time"] < 8].describe())
    # print(df_emb_layer.loc[df_emb_layer["time"] < 3.0].describe())
    print(df_emb_layer.describe())

    print(f"post process len: {df_post_process.size}")
    # print(f"post process avg: {df_post_process.mean()}")
    print(df_post_process.describe())
    
    print(f"{df_emb_layer.mean(numeric_only=True)},{df_layer.mean(numeric_only=True)},  {df_post_process.mean(numeric_only=True)}")
    


def get_layer_time(path, sign_array, num_mb):
    path = path
    sign_array = sign_array
    num_mb = num_mb
    layer_times = []
    embedding_times = []
    post_process = []
    with open(path) as f:
        lines = f.readlines()
        step_count = 0
        layer_count = 0
        for l in lines:
            if sign_array[0] in l: # embedding
                step_count = step_count + 1
                # print(step_count)
            if step_count > 40 * int(num_mb):
                if sign_array[0] in l:# embedding
                    # print(l.split())
                    embedding_time = float(l.split()[3])
                    embedding_times.append(embedding_time)
                    # print(embedding_time)
                if sign_array[1] in l:# Layer time
                    # print(l)
                    layer_time = float(l.split()[3])
                    layer_times.append(layer_time)     
                    # print(layer_time)
                if sign_array[2] in l: # post process
                    # print(l.split())
                    post_process_time = float(l.split()[2])
                    post_process.append(post_process_time)
    return embedding_times, layer_times, post_process

def get_decoder_time(path, sign_array):
    path = path
    sign_array = sign_array
    decoder_times = []
    with open(path) as f:
        lines = f.readlines()
        step_count = 1
        layer_count = 0
        
        for l in lines:
            if sign_array[1] in l: # global step
                step_count = step_count + 1
                # print(l)
                # print(step_count)
            if step_count > 40 :
                if sign_array[0] in l: # decoder time
                    # print(l)
                    decoder_time = float(l.split()[2])
                    decoder_times.append(decoder_time)
    return decoder_times
    
    
if __name__ == '__main__':
    main()
    
