import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import importlib.util
import os
import argparse
import vllm.envs as envs
import ray
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
import pickle
from math import comb
from ray.runtime_env import RuntimeEnv
import time

# envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"

def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--surround_with_messages", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_gpus", default=4, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    # parser.add_argument("--use_qwen_check", action="store_true")
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy 
    print(f"current stop list: {args.stop}")
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format

@ray.remote(num_gpus=1)
class InferenceGPU:
    def __init__(self, model_name_or_path, sampling_params, tokenizer, args, gpu_id):
        # 设置随机种子
        set_seed(args.seed)  # 设置Python的random和numpy的随机种子
        torch.manual_seed(args.seed)  # 设置PyTorch的随机种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["MASTER_PORT"] = str(29300 + gpu_id)
        print(f"VLLMWorker initializing on GPU {gpu_id}")
        self.model_name_or_path = model_name_or_path
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.args = args
        self.llm = LLM(model=self.model_name_or_path,
                       tensor_parallel_size=1,
                       trust_remote_code=True,
                       gpu_memory_utilization=0.96,
                       dtype="auto")
        print(f"VLLMWorker initialized on GPU {gpu_id}")

    
    def generate(self, examples_batch, generation_epoch):
        file_outputs_batch = []
        # Prepare all prompts in a single batch
        # Now generate responses for the entire batch in a single call
        for cur_generation_epoch in range(generation_epoch):
            completions = self.llm.generate(examples_batch, self.sampling_params)
            
            for i in range(len(completions)):
                generated_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]

                if cur_generation_epoch == 0:
                    # For each example in the batch, process the response
                    file_outputs_batch.append({
                        "question": examples_batch[i],
                        "generated_responses": generated_responses,
                    })
                else:
                    file_outputs_batch[i]['generated_responses'] += generated_responses

        return file_outputs_batch


def infer(args):
    num_gpus = args.num_gpus
    base_vllm_port = int(os.getenv("BASE_VLLM_PORT", "43543"))
    base_master_port = int(os.getenv("BASE_MASTER_PORT", "29400"))
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    n_sampling = args.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = n_sampling // factor
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")

    sampling_params = SamplingParams(temperature=args.temperature, 
                                     max_tokens=args.max_tokens, 
                                     n=factor,
                                     top_p=args.top_p
                                     )
    
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]
    
    out_file_prefix = f'{args.split}_{args.prompt_type}_t{args.temperature}_len{args.max_tokens}'
    out_file = f'{args.output_dir}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'
    
    os.makedirs(f'{args.output_dir}/{args.data_name}', exist_ok=True)
    os.makedirs(f'{args.completions_save_dir}/{args.data_name}', exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    workers = []
    for gpu_id in range(args.num_gpus):
        # 为每个worker分配不同的端口
        vllm_port = base_vllm_port + (gpu_id * 2)  # 每个GPU间隔2个端口号
        master_port = base_master_port + gpu_id
        
        worker = InferenceGPU.options(
            num_gpus=1,
            runtime_env=RuntimeEnv(
                env_vars={
                    "VLLM_PORT": str(vllm_port),
                    "MASTER_PORT": str(master_port)
                }
            )
        ).remote(model_name_or_path, sampling_params, tokenizer, args, gpu_id)
        workers.append(worker)

    ray.get([worker.__ray_ready__.remote() for worker in workers])
    
    
    prompt_batch = []
    for example in tqdm(examples, total=len(examples)):
        question = parse_question(example, args.data_name)
        system_prompt, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)
        
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)
        if args.surround_with_messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt}
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cur_prompt)
    print(prompt_batch[0])

    total_size = len(prompt_batch)
    num_workers = len(workers)
    batch_size = (total_size + num_workers - 1) // num_workers  # 向上取整
        
    # 记录每个chunk的起始索引
    chunk_indices = []
    chunks = []
    for i in range(num_workers):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_size)
        if start_idx < total_size:
            chunk_indices.append((start_idx, end_idx))
            chunks.append(prompt_batch[start_idx:end_idx])
    
    # Submit the inference tasks in parallel across multiple GPUs
    futures = []
    for i, example_batch in enumerate(chunks):
        worker = workers[i]
        futures.append((chunk_indices[i],worker.generate.remote(example_batch, generation_epoch)))
    
    # Collect the results
    start_time = time.time()
    print(f"Start inference at {start_time}")
    file_outputs = []
    for (start_idx, end_idx), future in tqdm(futures, "Waiting for inference results..."):
        file_outputs_batch = ray.get(future)
        file_outputs.append((start_idx, file_outputs_batch))
    file_outputs.sort(key=lambda x: x[0])  # 按起始索引排序
    end_time = time.time()
    print(f"End inference at {end_time}")
    delta_time = end_time - start_time
    print(f"Total time: {delta_time:.2f} seconds")

    file_outputs_new = []
    for _, texts in file_outputs:
        file_outputs_new.extend(texts)
    
    # Post-process the results
    pass_at_k_list = []
    k = args.k
    correct_cnt = 0

    for i in tqdm(range(len(examples)), "check correct..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
        generated_responses =file_outputs_new[i]['generated_responses']
        generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
        is_correct_list = [check_is_correct(generated_answer, gt_ans, args.data_name) for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1
        file_outputs_new[i]['generated_answers'] = generated_answers
        file_outputs_new[i]['gold_answer'] = gt_ans
        file_outputs_new[i]['is_correct'] = is_correct
        file_outputs_new[i]['answers_correctness'] = is_correct_list
        
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)
    
    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, 'w', encoding='utf-8') as f:
        count = 0
        for d in tqdm(file_outputs_new, "writing generation to jsonl file..."):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 100 == 0:
                f.flush()
        f.flush()
    os.rename(temp_out_file, out_file)
    
    print(f"correct cnt / total cnt: {correct_cnt}/{len(examples)}")
    print(f"Acc: {correct_cnt / len(examples):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"Pass@1: {correct_cnt}/{len(examples)} = {correct_cnt / len(examples):.4f}")

if __name__ == "__main__":
    args = parse_args()
    
    # 设置所有可能的随机种子
    set_seed(args.seed)  # 这会设置Python的random和numpy的随机种子
    
    # 设置PyTorch的随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 通过环境变量设置Ray的随机种子
    os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"  # 禁用内存监控以减少随机性
    os.environ["RAY_DEDUP_LOGS"] = "0"  # 禁用日志去重以减少随机性
    
    # 初始化Ray
    ray.init(num_cpus=args.num_gpus)
    
    # Running inference
    infer(args)
    
    # Shutdown Ray at the end
    ray.shutdown()
