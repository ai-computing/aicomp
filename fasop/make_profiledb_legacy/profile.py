import numpy as np
import os
import argparse

class Profile():
    def __init__(self, model_name, gpu_type, 
                 input_embedding_time=[], encoder_time=[], post_process_time=[], 
                 decoder_embedding_time=[], decoder_time=[], decoder_post_process_time=[], 
                 en_layer_num=0, de_layer_num=0, transformer_type=None, add_name=None, dir_path=None):
        self.model_name = model_name
        self.gpu_type = gpu_type
        
        # self.xxx_time = array[TP=1, TP=2, TP=4]
        
        self.input_embedding_time = input_embedding_time
        # print(type(self.input_embedding_time))
        self.encoder_time = encoder_time
        self.post_process_time = post_process_time
        self.decoder_embedding_time = decoder_embedding_time
        self.decoder_time = decoder_time
        self.decoder_post_process_time = decoder_post_process_time
        self.en_layer_num = en_layer_num
        self.de_layer_num = de_layer_num
        self.profile_array = []
        self.tp_array = ['1','2','4']
        self.transformer_type = transformer_type
        self.add_name = add_name
        
        if self.add_name is not None: 
            self.add_name = "_" + add_name 
        else :
            self.add_name = ""
        self.dir_path = dir_path
        # is_model()
    
    def is_model(self):
        
        if self.model_name == 'T5':
            assert self.en_layer_num > 0 and self.de_layer_num > 0
            return "both"
        elif self.model_name == 'gpt2XL':
            assert self.de_layer_num > 0
            return "de"
        elif self.model_name == 'bert':
            assert self.en_layer_num > 0
            return "en"
        else:
            assert False
    
    def make_nparray(self, iet, et, ppt, det, dt, dppt):
        
        # iet: input_embedding_time
        # et: encoder_time
        # pt: post_process_time
        # det: decoder_embedding_time
        # dt: decoder_time
        # dppt: decoder_post_process_time

        if self.transformer_type in ["en", "both"]:
            # pre process    
            self.profile_array.append(iet)
            # encoder
            for i in range(self.en_layer_num):
                self.profile_array.append(et)
            # post process    
            self.profile_array[-1] += ppt
        
        if self.transformer_type in ["de", "both"]:
            # pre process(=embedding)
            self.profile_array.append(det)
            # decoder
            for i in range(self.de_layer_num):
                self.profile_array.append(dt)
            # post process    
            if self.model_name == "T5":
                self.profile_array[-1] += dppt
            elif self.model_name == "gpt2XL":
                self.profile_array.append(dppt)
        return self.profile_array
        
    def save_nparray(self):
        self.transformer_type = self.is_model()
        
        for i in range(len(self.tp_array)):
            self.profile_array = []
            if self.transformer_type in ["en"]:
                iet = self.input_embedding_time[i]
                et = self.encoder_time[i]
                ppt = self.post_process_time[i]
                det = 0
                dt = 0
                dppt = 0
            elif self.transformer_type in ["de"]:
                iet = 0
                et = 0
                ppt = 0
                det = self.decoder_embedding_time[i]
                dt = self.decoder_time[i]
                dppt = self.decoder_post_process_time[i]
            elif self.transformer_type in ["both"]:
                iet = self.input_embedding_time[i]
                et = self.encoder_time[i]
                ppt = self.post_process_time[i]
                det = self.decoder_embedding_time[i]
                dt = self.decoder_time[i]
                dppt = self.decoder_post_process_time[i]
            
            nparray = self.make_nparray(iet, et, ppt, det, dt, dppt)
            filename = self.model_name +"_"+ self.gpu_type + "_" + self.tp_array[i] + self.add_name + ".npy"
            np.save(f'./{filename}', nparray)
    
    def print_nparray(self):
        for i in range(len(self.tp_array)):
            ar = np.load(f"./{self.model_name}_{self.gpu_type}_{self.tp_array[i]}" + self.add_name + ".npy")
            print(f"./{self.model_name}_{self.gpu_type}_{self.tp_array[i]}" + self.add_name + ".npy")
            print(f"size: {len(ar)}")
            print(ar)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gpt2XL, bert, T5)')
    parser.add_argument('--gpu_type', type=str, required=True, help='GPU type (e.g., A10, A100)')
    parser.add_argument('--transformer_type', type=str, required=True, choices=['en', 'de', 'both'], help='Transformer type')
    parser.add_argument('--add_name', type=str, help='Additional name for the output file')

    # Encoder arguments
    parser.add_argument('--input_embedding_time', nargs='+', type=float, help='Input embedding time (TP=1, 2, 4)')
    parser.add_argument('--encoder_time', nargs='+', type=float, help='Encoder time (TP=1, 2, 4)')
    parser.add_argument('--post_process_time', nargs='+', type=float, help='Post process time (TP=1, 2, 4)')
    parser.add_argument('--en_layer_num', type=int, help='Number of encoder layers')

    # Decoder arguments
    parser.add_argument('--decoder_embedding_time', nargs='+', type=float, help='Decoder embedding time (TP=1, 2, 4)')
    parser.add_argument('--decoder_time', nargs='+', type=float, help='Decoder time (TP=1, 2, 4)')
    parser.add_argument('--decoder_post_process_time', nargs='+', type=float, help='Decoder post process time (TP=1, 2, 4)')
    parser.add_argument('--de_layer_num', type=int, help='Number of decoder layers')

    args = parser.parse_args()

    if args.transformer_type in ['en', 'both']:
        if not all([args.input_embedding_time, args.encoder_time, args.post_process_time, args.en_layer_num]):
            print("Error: For 'en' or 'both' transformer type, you must provide input_embedding_time, encoder_time, post_process_time, and en_layer_num.")
            parser.print_usage()
            exit()

    if args.transformer_type in ['de', 'both']:
        if not all([args.decoder_embedding_time, args.decoder_time, args.decoder_post_process_time, args.de_layer_num]):
            print("Error: For 'de' or 'both' transformer type, you must provide decoder_embedding_time, decoder_time, decoder_post_process_time, and de_layer_num.")
            parser.print_usage()
            exit()

    profile = Profile(args.model_name, args.gpu_type,
                         transformer_type=args.transformer_type,
                         input_embedding_time=args.input_embedding_time,
                         encoder_time=args.encoder_time,
                         post_process_time=args.post_process_time,
                         en_layer_num=args.en_layer_num,
                         decoder_embedding_time=args.decoder_embedding_time,
                         decoder_time=args.decoder_time,
                         decoder_post_process_time=args.decoder_post_process_time,
                         de_layer_num=args.de_layer_num,
                         add_name=args.add_name)

    profile.save_nparray()
    profile.print_nparray()

    
if __name__ == "__main__":
    main()


