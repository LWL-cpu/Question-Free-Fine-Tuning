import os
import json
import argparse
from transformers import AutoTokenizer

def analyze_model_outputs(eval_results_dir, output_file, model_path, verbose=False):
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 用于存储所有数据集的汇总统计
    all_correct_token_lengths = []
    all_incorrect_token_lengths = []
    all_token_counts = []
    all_wait_counts = []
    all_correctness_list = []
    all_wait_zero_tokens = []
    all_wait_zero_correctness = []
    all_wait_greater_one_tokens = []
    all_wait_greater_one_correctness = []

    print(f"\nAnalyzing model outputs from: {eval_results_dir}", file=output_file)

    # 遍历根目录下的每个子目录
    for subdir in os.listdir(eval_results_dir):
        subdir_path = os.path.join(eval_results_dir, subdir)
        
        # 检查是否是子目录
        if os.path.isdir(subdir_path):
            # 查找 jsonl 文件
            for file_name in os.listdir(subdir_path):
                if file_name.endswith('.jsonl'):
                    file_path = os.path.join(subdir_path, file_name)
                    
                    # 为每个数据集初始化统计变量
                    correct_token_lengths = []
                    incorrect_token_lengths = []
                    threshold = 32 * 1024
                    token_counts = []
                    wait_counts = []
                    correctness_list = []
                    wait_zero_tokens = []
                    wait_zero_correctness = []
                    wait_greater_one_tokens = []
                    wait_greater_one_correctness = []
                    
                    total_token_count = 0
                    total_items = 0
                    small_token_count = 0
                    small_token_total = 0
                    large_token_count = 0
                    large_token_total = 0
                    
                    # 打开并读取 jsonl 文件
                    with open(file_path, 'r') as f:
                        print(f"\nDetailed Solution Analysis for Dataset {subdir}/{file_name}:", file=output_file)
                        print("-" * 50, file=output_file)
                        for line in f:
                            item = json.loads(line)
                            generated_responses = item.get('generated_responses', [])
                            answers_correctness = item.get('answers_correctness', [])
                            
                            for response, correctness in zip(generated_responses, answers_correctness):
                                tokens = tokenizer.tokenize(response)
                                token_count = len(tokens)
                                
                                wait_count = response.lower().count('wait')
                                al_count = response.lower().count('alternatively')
                                
                                token_counts.append(token_count)
                                wait_counts.append(wait_count)
                                correctness_list.append(correctness)
                                
                                if wait_count > 3 or (wait_count > 0 and al_count > 0):
                                    wait_greater_one_tokens.append(token_count)
                                    wait_greater_one_correctness.append(correctness)
                                else:
                                    wait_zero_tokens.append(token_count)
                                    wait_zero_correctness.append(correctness)
                                
                                if token_count < threshold:
                                    small_token_count += 1
                                    small_token_total += token_count
                                
                                if token_count > threshold:
                                    large_token_count += 1
                                    large_token_total += token_count
                                
                                if correctness:
                                    correct_token_lengths.append(token_count)
                                else:
                                    incorrect_token_lengths.append(token_count)
                                
                                total_token_count += token_count
                                total_items += 1
                    
                    # 打印数据集统计信息
                    if total_items > 0:
                        avg_token_count = total_token_count / total_items
                        
                        if verbose:
                            small_token_ratio = (small_token_count / total_items) * 100
                            avg_small_token = small_token_total / small_token_count if small_token_count > 0 else 0
                            large_token_ratio = (large_token_count / total_items) * 100
                            avg_large_token = large_token_total / large_token_count if large_token_count > 0 else 0
                            
                            print(f"Summary for Dataset: {subdir}/{file_name}", file=output_file)
                            print(f"  Average Token Count: {avg_token_count:.2f}", file=output_file)
                            print(f"  Token Length < 16K Proportion: {small_token_ratio:.2f}%", file=output_file)
                            print(f"  Average Token Length (< 16K): {avg_small_token:.2f}", file=output_file)
                            print(f"  Token Length > 16K Proportion: {large_token_ratio:.2f}%", file=output_file)
                            print(f"  Average Token Length (> 16K): {avg_large_token:.2f}", file=output_file)
                        else:
                            print(f"{subdir}/{file_name}: {avg_token_count:.2f}", file=output_file)
                    
                    # 打印 wait_count 统计信息（仅在详细模式下）
                    if verbose:
                        print(f"\nWait Count Statistics for Dataset {subdir}/{file_name}:", file=output_file)
                        print("-" * 50, file=output_file)
                        
                        # wait_count == 0
                        if wait_zero_tokens:
                            wait_zero_count = len(wait_zero_tokens)
                            avg_wait_zero_length = sum(wait_zero_tokens) / wait_zero_count
                            avg_wait_zero_correctness = sum(wait_zero_correctness) / wait_zero_count * 100
                        else:
                            wait_zero_count = 0
                            avg_wait_zero_length = 0
                            avg_wait_zero_correctness = 0
                        print(f"Points with wait_count == 0:", file=output_file)
                        print(f"  Number of Points: {wait_zero_count}", file=output_file)
                        print(f"  Average Token Length: {avg_wait_zero_length:.2f}", file=output_file)
                        print(f"  Average Correctness: {avg_wait_zero_correctness:.2f}%", file=output_file)
                        
                        # wait_count > 0
                        if wait_greater_one_tokens:
                            wait_greater_one_count = len(wait_greater_one_tokens)
                            avg_wait_greater_one_length = sum(wait_greater_one_tokens) / wait_greater_one_count
                            avg_wait_greater_one_correctness = sum(wait_greater_one_correctness) / wait_greater_one_count * 100
                        else:
                            wait_greater_one_count = 0
                            avg_wait_greater_one_length = 0
                            avg_wait_greater_one_correctness = 0
                        print(f"Points with wait_count > 0:", file=output_file)
                        print(f"  Number of Points: {wait_greater_one_count}", file=output_file)
                        print(f"  Average Token Length: {avg_wait_greater_one_length:.2f}", file=output_file)
                        print(f"  Average Correctness: {avg_wait_greater_one_correctness:.2f}%", file=output_file)
                        print("-" * 50, file=output_file)
                    
                    # 合并到总统计
                    all_correct_token_lengths.extend(correct_token_lengths)
                    all_incorrect_token_lengths.extend(incorrect_token_lengths)
                    all_token_counts.extend(token_counts)
                    all_wait_counts.extend(wait_counts)
                    all_correctness_list.extend(correctness_list)
                    all_wait_zero_tokens.extend(wait_zero_tokens)
                    all_wait_zero_correctness.extend(wait_zero_correctness)
                    all_wait_greater_one_tokens.extend(wait_greater_one_tokens)
                    all_wait_greater_one_correctness.extend(wait_greater_one_correctness)

    # 打印总体统计信息（仅在详细模式下）
    if verbose:
        print("\nOverall Wait Count Statistics:", file=output_file)
        print("-" * 50, file=output_file)
        
        # wait_count == 0
        if all_wait_zero_tokens:
            all_wait_zero_count = len(all_wait_zero_tokens)
            all_avg_wait_zero_length = sum(all_wait_zero_tokens) / all_wait_zero_count
            all_avg_wait_zero_correctness = sum(all_wait_zero_correctness) / all_wait_zero_count * 100
        else:
            all_wait_zero_count = 0
            all_avg_wait_zero_length = 0
            all_avg_wait_zero_correctness = 0
        print(f"Points with wait_count == 0:", file=output_file)
        print(f"  Number of Points: {all_wait_zero_count}", file=output_file)
        print(f"  Average Token Length: {all_avg_wait_zero_length:.2f}", file=output_file)
        print(f"  Average Correctness: {all_avg_wait_zero_correctness:.2f}%", file=output_file)

        # wait_count > 0
        if all_wait_greater_one_tokens:
            all_wait_greater_one_count = len(all_wait_greater_one_tokens)
            all_avg_wait_greater_one_length = sum(all_wait_greater_one_tokens) / all_wait_greater_one_count
            all_avg_wait_greater_one_correctness = sum(all_wait_greater_one_correctness) / all_wait_greater_one_count * 100
        else:
            all_wait_greater_one_count = 0
            all_avg_wait_greater_one_length = 0
            all_avg_wait_greater_one_correctness = 0
        print(f"Points with wait_count > 0:", file=output_file)
        print(f"  Number of Points: {all_wait_greater_one_count}", file=output_file)
        print(f"  Average Token Length: {all_avg_wait_greater_one_length:.2f}", file=output_file)
        print(f"  Average Correctness: {all_avg_wait_greater_one_correctness:.2f}%", file=output_file)
        print("-" * 50, file=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model outputs')
    parser.add_argument('eval_dir', type=str, help='Path to the evaluation results directory')
    parser.add_argument('--model_path', type=str, help='Path to the tokenizer model')
    parser.add_argument('--output_file', type=str, default='eval_results/analysis.txt', help='Name of output analysis file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output mode (default: concise mode)')
    args = parser.parse_args()

    eval_results_dir = args.eval_dir
    # 确保输出目录存在
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # 设置分析结果文件路径
    output_file_path = os.path.join(eval_results_dir, args.output_file)
    
    print(f"Processing model outputs from: {eval_results_dir}")
    print(f"Tokenizer path: {args.model_path}")
    print(f"Output mode: {'Verbose' if args.verbose else 'Concise'}")
    print(f"\033[92mResults will be saved to: {output_file_path}\033[0m")
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        analyze_model_outputs(eval_results_dir, output_file, args.model_path, args.verbose) 
