#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, July 24th 2023, 11:59:49 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

XLLM_PATH = '/home/project/13003565/geyu/x-LLM'

MODEL_PATH = {

    'random' : 'random',
    
    'alpaca-7b': '../prepared_models/alpaca-7b',
    
    'flan-t5-small': '../prepared_models/flan-t5-small',
    'flan-t5-base' : '../prepared_models/flan-t5-base',
    'flan-t5-large': '../prepared_models/flan-t5-large',
    'flan-t5-xl'   : '../prepared_models/flan-t5-xl',
    'flan-t5-xxl'  : '../prepared_models/flan-t5-xxl',
    'flan-ul2'     : '../prepared_models/flan-ul2',

    'vicuna-7b' : '../prepared_models/vicuna-7b-v1.3',
    'vicuna-13b': '../prepared_models/vicuna-13b-v1.3',
    'vicuna-33b': '../prepared_models/vicuna-33b-v1.3',

    'llama-7b' : '../prepared_models/llama-7b',
    'llama-13b': '../prepared_models/llama-13b',
    'llama-30b': '../prepared_models/llama-30b',
    'llama-65b': '../prepared_models/llama-65b',

    'llama-2-7b'      : '/home/project/13003565/llama-2-7b',
    'llama-2-7b-chat' : '../prepared_models/llama-2-7b-chat',
    'llama-2-13b'     : '../prepared_models/llama-2-13b',
    'llama-2-13b-chat': '../prepared_models/llama-2-13b-chat',
    'llama-2-70b'     : '../prepared_models/llama-2-70b',
    'llama-2-70b-chat': '../prepared_models/llama-2-70b-chat',

    'chatglm-6b' : '../prepared_models/chatglm-6b',
    'chatglm2-6b': '../prepared_models/chatglm2-6b',
    'chatglm3-6b': '../prepared_models/chatglm3-6b',

    'baichuan-7b'      : '../prepared_models/baichuan-7b',
    'baichuan-13b'     : '../prepared_models/baichuan-13b',
    'baichuan-13b-chat': '../prepared_models/baichuan-13b-chat',

    'chatgpt' : 'gpt-3.5-turbo-0613',
    'chatgpt4': 'gpt-4-0613',

    'seallama-13b-220823': '../prepared_models/seallama-220823',
    'seallama-7b-040923' : '../prepared_models/seallama-040923',

    'baichuan-2-7b'      : '../prepared_models/baichuan-2-7b',
    'baichuan-2-7b-chat' : '../prepared_models/baichuan-2-7b-chat',
    'baichuan-2-13b'     : '../prepared_models/baichuan-2-13b',
    'baichuan-2-13b-chat': '../prepared_models/baichuan-2-13b-chat',

    'vicuna-7b-v1.5' : '../prepared_models/vicuna-7b-v1.5',
    'vicuna-13b-v1.5': '../prepared_models/vicuna-13b-v1.5',

    'bloomz-7b1' : '../prepared_models/bloomz-7b1',
    'mt0-xxl'    : '../prepared_models/mt0-xxl',

    'colossal-llama-2-7b-base' : '../prepared_models/colossal-llama-2-7b-base',

    'fastchat-t5-3b-v1.0': '../prepared_models/fastchat-t5-3b-v1.0',

    # x-llm
    "llama-2-7b-hf.alpaca_en+alpaca_es.finetune": f"{XLLM_PATH}/model/llama-2-7b-hf.alpaca_en+alpaca_es.finetune",
    "llama-2-7b-hf.alpaca_en+alpaca_es+translation_ncwm_en-es.finetune": f"{XLLM_PATH}/model/llama-2-7b-hf.alpaca_en+alpaca_es+translation_ncwm_en-es.finetune",
    "llama-2-7b-hf": f"{XLLM_PATH}/model/llama-2-7b-hf",
    "llama-2-7b-hf.alpaca_en+alpaca_vi+alpaca_es+alpaca_zh.finetune": f"{XLLM_PATH}/model/llama-2-7b-hf.alpaca_en+alpaca_vi+alpaca_es+alpaca_zh.finetune",
    "llama-2-7b-hf.alpaca_en.finetune": f"{XLLM_PATH}/model/llama-2-7b-hf.alpaca_en.finetune",
    "llama-2-7b-hf.alpaca_en+alpaca_vi+translation_ncwm_en-vi.finetune": f"{XLLM_PATH}/model/llama-2-7b-hf.alpaca_en+alpaca_vi+translation_ncwm_en-vi.finetune",
    "llama-2-7b-hf.alpaca_en+alpaca_zh.finetune": f"{XLLM_PATH}/model/llama-2-7b-hf.alpaca_en+alpaca_zh.finetune",
    "llama-2-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune": f"{XLLM_PATH}/model/llama-2-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune",
    "llama-2-7b-hf.alpaca_en+alpaca_vi.finetune": f"{XLLM_PATH}/model/llama-2-7b-hf.alpaca_en+alpaca_vi.finetune",
    
    # sharegpt
    "llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_es+sharegpt-clean_zh+sharegpt-clean_vi.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/sharegpt-cps/llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_es+sharegpt-clean_zh+sharegpt-clean_vi.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_zh.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/sharegpt-cps/llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_zh.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.sharegpt-clean_en.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/encomp-cps/llama-2-7b-hf.sharegpt-clean_en.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_vi.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/sharegpt-cps/llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_vi.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_es.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/sharegpt-cps/llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_es.finetune.general_prompt_no_sys",
    # alpaca-gpt4
 "llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_es.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/alpaca-gpt4-cps/llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_es.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.alpaca-gpt4_en.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/encomp-cps/llama-2-7b-hf.alpaca-gpt4_en.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_zh.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/alpaca-gpt4-cps/llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_zh.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_es+alpaca-gpt4_zh+alpaca-gpt4_vi.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/alpaca-gpt4-cps/llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_es+alpaca-gpt4_zh+alpaca-gpt4_vi.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_vi.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/alpaca-gpt4-cps/llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_vi.finetune.general_prompt_no_sys",
    #platypus
 "llama-2-7b-hf.platypus_en+platypus_vi+platypus_zh+platypus_es.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/platypus-cps/llama-2-7b-hf.platypus_en+platypus_vi+platypus_zh+platypus_es.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.platypus_en+platypus_es.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/platypus-cps/llama-2-7b-hf.platypus_en+platypus_es.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.platypus_en.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/encomp-cps/llama-2-7b-hf.platypus_en.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.platypus_en+platypus_zh.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/platypus-cps/llama-2-7b-hf.platypus_en+platypus_zh.finetune.general_prompt_no_sys",
 "llama-2-7b-hf.platypus_en+platypus_vi.finetune.general_prompt_no_sys": "/data/projects/13003565/geyu/x-LLM/model/platypus-cps/llama-2-7b-hf.platypus_en+platypus_vi.finetune.general_prompt_no_sys"
 
    
}


MODEL_LANG = {

    'random': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'alpaca-7b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'flan-t5-small': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'flan-t5-base': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'flan-t5-large': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'flan-t5-xl': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'flan-t5-xxl': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'flan-ul2': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'vicuna-7b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'vicuna-13b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'vicuna-33b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-7b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-13b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-30b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-65b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-2-7b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-2-7b-chat': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-2-13b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-2-13b-chat': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-2-70b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'llama-2-70b-chat': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'chatglm-6b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'chatglm2-6b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'chatglm3-6b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'baichuan-7b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'baichuan-13b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'chatgpt': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'chatgpt4': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'seallama-13b-220823': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'seallama-7b-040923': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'baichuan-2-7b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'baichuan-2-7b-chat': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'baichuan-2-13b': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'baichuan-2-13b-chat': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'vicuna-7b-v1.5': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],


    'vicuna-13b-v1.5': 
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'bloomz-7b1':
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'mt0-xxl':
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'colossal-llama-2-7b-base':
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    'fastchat-t5-3b-v1.0':
                [
                    'english',
                    'chinese',
                    'indonesian',
                    'spanish',
                    'vietnamese',
                    'malay',
                    'filipino',
                ],

    "llama-2-7b-hf.alpaca_en+alpaca_es.finetune": ['english', 'chinese', 'vietnamese', 'spanish'],
    "llama-2-7b-hf.alpaca_en+alpaca_es+translation_ncwm_en-es.finetune": ['english', 'chinese', 'vietnamese', 'spanish'],
    "llama-2-7b-hf": ['english', 'chinese', 'vietnamese', 'spanish'],
    "llama-2-7b-hf.alpaca_en+alpaca_vi+alpaca_es+alpaca_zh.finetune": ['english', 'chinese', 'vietnamese', 'spanish'],
    "llama-2-7b-hf.alpaca_en.finetune": ['english', 'chinese', 'vietnamese', 'spanish'],
    "llama-2-7b-hf.alpaca_en+alpaca_vi+translation_ncwm_en-vi.finetune": ['english', 'chinese', 'vietnamese', 'spanish'],
    "llama-2-7b-hf.alpaca_en+alpaca_zh.finetune": ['english', 'chinese', 'vietnamese', 'spanish'],
    "llama-2-7b-hf.alpaca_en+alpaca_zh+translation_ncwm_en-zh.finetune": ['english', 'chinese', 'vietnamese', 'spanish'],
    "llama-2-7b-hf.alpaca_en+alpaca_vi.finetune": ['english', 'chinese', 'vietnamese', 'spanish'],
    
"llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_es.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.alpaca-gpt4_en.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_zh.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_es+alpaca-gpt4_zh+alpaca-gpt4_vi.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.alpaca-gpt4_en+alpaca-gpt4_vi.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],

"llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_es+sharegpt-clean_zh+sharegpt-clean_vi.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_zh.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.sharegpt-clean_en.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_vi.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.sharegpt-clean_en+sharegpt-clean_es.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],

"llama-2-7b-hf.platypus_en+platypus_vi+platypus_zh+platypus_es.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.platypus_en+platypus_es.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.platypus_en.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.platypus_en+platypus_zh.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish'],
"llama-2-7b-hf.platypus_en+platypus_vi.finetune.general_prompt_no_sys": ['english', 'chinese', 'vietnamese', 'spanish']
}
