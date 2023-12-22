#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, July 25th 2023, 10:11:14 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import os

# add parent directory to sys.path
import sys
sys.path.append('.')

import time
import logging

from typing import List

import openai
import torch
import transformers

from config.model_path import (
    MODEL_PATH,
)

CHATGPT_RETRY_INTERVAL = 20
CHATGPT_RETRY_TIMES    = 10
CHATGPT_ERROR_OUTPUT   = "#ERROR#"




# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 



class Model(object):

    def __init__(self, model_name_or_path, max_new_tokens=128):
        
        self.model_name     = model_name_or_path
        self.max_new_tokens = max_new_tokens

        self.load_model()

    def load_model(self):

        if self.model_name in MODEL_PATH:
            self.model_path = MODEL_PATH[self.model_name]
            logger.info("Loading model: {}".format(self.model_path))
        else:
            # raise NotImplementedError("Model {} not implemented yet".format(self.model_name))
            logger.info("Module not exists, loadding from path directly")
            self.model_path = self.model_name
            self.model_name = os.path.basename(self.model_path)
            if not os.path.exists(self.model_path):
                logger.error("Model path is not correct")
                raise NotImplementedError("Model {} not exists yet".format(self.model_name)) 

        
        if self.model_name in ['alpaca-7b', 
                               'vicuna-7b', 
                               'vicuna-13b',
                               'vicuna-33b',
                               'vicuna-7b-v1.5',
                               'vicuna-13b-v1.5',
                               'llama-7b',
                               'llama-13b',
                               'llama-30b',
                               'llama-65b',
                               'llama-2-7b',
                               'llama-2-7b-chat',
                               'llama-2-13b',
                               'llama-2-13b-chat',
                               'llama-2-70b',
                               'llama-2-70b-chat',
                               'seallama-13b-220823',
                               'seallama-7b-040923',
                               'bloomz-7b1',
                               
                               'llama-2-7b-hf.alpaca_en+alpaca_es.finetune', 
                               'llama-2-7b-hf.alpaca_en+alpaca_es+translation_ncwm_en-es.finetune', 
                               'llama-2-7b-hf', 
                               'llama-2-7b-hf.alpaca_en+alpaca_vi+alpaca_es+alpaca_zh.finetune', 
                               'llama-2-7b-hf.alpaca_en.finetune', 
                               'llama-2-7b-hf.alpaca_en+alpaca_vi+translation_ncwm_en-vi.finetune', 
                               'llama-2-7b-hf.alpaca_en+alpaca_zh.finetune', 
                               'llama-2-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune', 
                               'llama-2-7b-hf.alpaca_en+alpaca_vi.finetune',
                               "llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_es.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.alpaca-gpt4_en.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_zh.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_es+alpaca-gpt4_zh+alpaca-gpt4_vi.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_vi.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_es+sharegpt-clean_zh+sharegpt-clean_vi.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_zh.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.sharegpt-clean_en.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_vi.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_es.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.platypus_en+platypus_vi+platypus_zh+platypus_es.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.platypus_en+platypus_es.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.platypus_en.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.platypus_en+platypus_zh.finetune.general_prompt_no_sys",
                                "llama-2-7b-hf.platypus_en+platypus_vi.finetune.general_prompt_no_sys"



                               ]:
            self._load_model_llama_family()
        

        elif self.model_name in ['flan-t5-small',
                                 'flan-t5-base',
                                 'flan-t5-large',
                                 'flan-t5-xl',
                                 'flan-t5-xxl',
                                 'flan-ul2',
                                 'fastchat-t5-3b-v1.0',
                                ]:
            self._load_t5()

        elif self.model_name in ['mt0-xxl'
                                 ]:
            self._load_mt0()

        elif self.model_name in ['chatglm-6b',
                                 'chatglm2-6b',
                                 'chatglm3-6b',
                                 ]:
            self._load_model_chatglm()

        elif self.model_name in ['baichuan-7b',
                                 'baichuan-13b',
                                 'baichuan-2-7b',
                                 'baichuan-2-13b'
                                 ]:
            self._load_model_baichuan()

        elif self.model_name in ['baichuan-13b-chat',
                                 'baichuan-2-7b-chat',
                                 'baichuan-2-13b-chat'
                                 ]:
            self._load_model_baichuan_chat()


        elif self.model_name in ['chatgpt', 
                                 'chatgpt4'
                                 ]:
            logger.info('Loading chatgpt as API calls')

        elif self.model_name in ['random']:
            logger.info('Loading random generation model')

        elif self.model_name in ['colossal-llama-2-7b-base']:
            self._load_model_colossal()

        else:
            # raise NotImplementedError("Model {} not implemented yet".format(self.model_name))
            self._load_model_llama_family() # default llama family
        
    
    def generate(self, batch_input, batch_tag):
        return self._generate_llama_family(batch_input, batch_tag)  # default llama family

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - Load Model  - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def _load_model_llama_family(self):
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, device_map="auto", use_fast=False, padding_side='left')

        # Load model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.bfloat16)
        self.model.eval() # set to eval mode, by default it is in eval model but just in case
        logger.info("Model loaded: {} in 16 bits".format(self.model_path))

        # Load Pad token
        logger.info('Tokenizer pad token = {}'.format(self.tokenizer.pad_token))
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<unk>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info('Added <unk> to the tokenizer {}'.format(self.model_path))


    def _load_t5(self):

        self.tokenizer = transformers.T5Tokenizer.from_pretrained(self.model_path, use_fast=False, device_map="auto")
        self.model     = transformers.T5ForConditionalGeneration.from_pretrained(self.model_path, device_map="auto")
        self.model.eval() # set to eval mode, by default it is in eval model but just in case
        logger.info("Model loaded: {} in 32 bits".format(self.model_path))


    def _load_mt0(self):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=False, device_map="auto")
        self.model     = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.model_path, device_map="auto")
        self.model.eval() # set to eval mode, by default it is in eval model but just in case
        logger.info("Model loaded: {} in 32 bits".format(self.model_path))


    def _load_model_chatglm(self):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=False, device_map="auto", trust_remote_code=True)
        self.model = transformers.AutoModel.from_pretrained(self.model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        self.model.eval() # set to eval mode, by default it is in eval model but just in case
        logger.info("Model loaded: {} in 16 bits".format(self.model_path))
        

    def _load_model_baichuan(self):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=False, device_map="auto", trust_remote_code=True, padding_side='left')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        self.model.eval() # set to eval mode, by default it is in eval model but just in case
        logger.info("Model loaded: {} in 16 bits".format(self.model_path))

        # Load Pad token
        if self.tokenizer.pad_token is None:
            #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.add_special_tokens({'pad_token': '<unk>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info('Added <unk> to the tokenizer as padding token {}'.format(self.model_path))


    def _load_model_baichuan_chat(self):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=False, device_map="auto", trust_remote_code=True, padding_side='left')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        self.model.generation_config = transformers.GenerationConfig.from_pretrained(self.model_path)
        self.model.eval() # set to eval mode, by default it is in eval model but just in case
        logger.info("Model loaded: {} in 16 bits".format(self.model_path))

        # Load Pad token
        if self.tokenizer.pad_token is None:
            #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.add_special_tokens({'pad_token': '<unk>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info('Added <unk> to the tokenizer as padding token {}'.format(self.model_path))


    def _load_model_colossal(self):

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=False, device_map="auto", trust_remote_code=True, padding_side='left')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        self.model.generation_config = transformers.GenerationConfig.from_pretrained(self.model_path)
        self.model.eval() # set to eval mode, by default it is in eval model but just in case
        logger.info("Model loaded: {} in 16 bits".format(self.model_path))

        # Load Pad token
        if self.tokenizer.pad_token is None:
            #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.add_special_tokens({'pad_token': '<unk>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info('Added <unk> to the tokenizer as padding token {}'.format(self.model_path))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - Generation  - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def _generate_llama_family(self, batch_input, batch_tag=None):

        generation_config                = self.model.generation_config
        generation_config.max_new_tokens = self.max_new_tokens
        formatted_batch_input = []
        for i in range(len(batch_input)):
            B_INST, E_INST, B_SYS, E_SYS = "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"
            SYS_PROMPT=B_SYS + '\n' + "You are a helpful assistent.\n" + E_SYS + '\n\n'
            if batch_tag is None:
                formatted_batch_input.extend([f"{B_INST} {SYS_PROMPT} {batch_input[i]} {E_INST}"])
            else:
                formatted_batch_input.extend([f"{B_INST} {batch_tag[i]} {SYS_PROMPT} {batch_input[i]} {E_INST} {batch_tag[i]}"])
        input_ids                        = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
      
        if generation_config.pad_token_id != self.tokenizer.pad_token_id:
            generation_config.pad_token_id = self.tokenizer.pad_token_id
            logger.warning("syncing generation config pad with tokenizer")
        
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, generation_config = generation_config)

        # remove the input_ids from the output_ids
        output_ids = output_ids[:, input_ids.shape[-1]:]
        outputs    = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return outputs


    def _generate_llama_2_chat(self, batch_input):

        formatted_batch_input = []
        for input in batch_input:
            dialog = [{"role": "user", "content": input}]
            B_INST, E_INST = "[INST]", "[/INST]"
            formatted_batch_input.extend([f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"])
        batch_input = formatted_batch_input

        generation_config                = self.model.generation_config
        generation_config.max_new_tokens = self.max_new_tokens
        #input_ids                        = self.tokenizer.encode(batch_input[0], bos=True, eos=False, return_tensors="pt", padding=True).to(self.model.device)
        input_ids                        = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, generation_config = generation_config)

        # remove the input_ids from the output_ids
        output_ids = output_ids[:, input_ids.shape[-1]:]
        outputs    = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return outputs


    def _generate_t5(self, batch_input):

        input_ids  = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)

        if self.model_name in ['fastchat-t5-3b-v1.0']: # set repetition penalty to 3 for fastchat
            output_ids = self.model.generate(input_ids, max_length=self.max_new_tokens, early_stopping=True, repetition_penalty=3.0)
        else:
            output_ids = self.model.generate(input_ids, max_length=self.max_new_tokens, early_stopping=True)

        with torch.no_grad():
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return outputs
    

    def _generate_mt0(self, batch_input):

        input_ids  = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        output_ids = self.model.generate(input_ids, max_length=self.max_new_tokens, early_stopping=True)
        with torch.no_grad():
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return outputs


    def _generate_chatgpt(self, batch_input):

        if len(batch_input) != 1:
            raise ValueError("Our ChatGPT only supports batch size 1")
        
        outputs = CHATGPT_ERROR_OUTPUT
        for _ in range(CHATGPT_RETRY_TIMES):

            try:
                response = openai.ChatCompletion.create(
                        model = self.model_path,
                        messages = [{"role": "user", "content": batch_input[0]}],
                        n=1,
                    )
                outputs = response["choices"][0]["message"]["content"]
                break
                
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(CHATGPT_RETRY_INTERVAL)         

        if outputs == CHATGPT_ERROR_OUTPUT:
            logging.info("CHATGPT API failed to generate response.")
        
        outputs = [outputs]

        return outputs
        

    def _generate_chatglm(self, batch_input):

        if len(batch_input) != 1:
            raise ValueError("Our ChatGLM only supports batch size 1")

        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, batch_input[0], history=[])

        return [response]


    def _generate_baichuan(self, batch_input):

        #if len(batch_input) != 1:
        #    raise ValueError("Our Baichuan only supports batch size 1")
        
        #input_ids = self.tokenizer(batch_input[0], return_tensors="pt").input_ids.to(self.model.device)
        input_ids  = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens, repetition_penalty=1.1)
        
        output_ids = output_ids[:, input_ids.shape[-1]:]
        #output    = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output     = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return output
        

    def _generate_baichuan_chat(self, batch_input):

        if len(batch_input) != 1:
            raise ValueError("Our Baichuan-chat implementation only supports batch size 1")
        
        #input_ids = self.tokenizer(batch_input[0], return_tensors="pt").input_ids.to(self.model.device)

        messages = []
        messages.append({"role": "user", "content": batch_input[0]})
        output = self.model.chat(self.tokenizer, messages)

        return [output]


    def _generate_colossal(self, batch_input):

        if len(batch_input) != 1:
            raise ValueError("Our Colossal only supports batch size 1")

        input_ids  = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, 
                                             max_new_tokens=self.max_new_tokens, 
                                             do_sample=True,
                                             top_k=50,
                                             top_p=0.95,
                                             num_return_sequences=1
                                             )
        
        output_ids = output_ids[:, input_ids.shape[-1]:]
        output     = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return output
