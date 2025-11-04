import os
import json
from concurrent.futures import ThreadPoolExecutor
import random
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from utils.parser import extract_answer
from utils.grader import check_is_correct

# 并行检查答案正确性
def parallel_check_is_correct(answers, gold_answer_parsed, dataset, max_workers=40):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda ans: check_is_correct(ans, gold_answer_parsed, dataset), answers))
    return results

# 计算major@k准确率
def major_vote_accuracy(generated_answers, gold_answer, k=4, num_samples=10):
    correct_count = 0
    gold_answer = gold_answer

    for _ in range(num_samples):
        sampled_answers = random.sample(generated_answers, k)
        answer_counts = {}
        for ans in sampled_answers:
            ans = ans
            if ans:
                answer_counts[ans] = answer_counts.get(ans, 0) + 1

        major_answer = max(answer_counts, key=answer_counts.get) if answer_counts else ""
        if str(major_answer) == str(gold_answer):
            correct_count += 1

    return correct_count / num_samples

# 并行计算pass@1, pass@16, major@16
def compute_pass_at1(base_dir, output_file, max_workers=16):
    print(f"\nProcessing directory: {base_dir}", file=output_file)
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        print(f"Dataset: {dataset}", file=output_file)

        for file in os.listdir(dataset_path):
            if file.endswith(".jsonl"):
                file_path = os.path.join(dataset_path, file)
                total_accuracy_at1 = 0
                total_accuracy_at16 = 0
                total_major_accuracy_at16 = 0
                total_items = 0

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)

                        generated_answers = item.get("generated_responses", [])
                        extracted_answers = [extract_answer(answer, dataset) for answer in generated_answers]
                        gold_answer = item.get("gold_answer", "")

                        gold_answer_parsed = gold_answer

                        if extracted_answers and gold_answer:
                            correctness_results = parallel_check_is_correct(extracted_answers, gold_answer_parsed, dataset, max_workers)

                            item_accuracy_at1 = sum(correctness_results) / len(correctness_results)
                            item_accuracy_at16 = 1 if any(correctness_results) else 0

                            major_accuracy_at16 = major_vote_accuracy(extracted_answers, gold_answer_parsed, k=len(correctness_results))
                        else:
                            item_accuracy_at1 = 0
                            item_accuracy_at16 = 0
                            major_accuracy_at16 = 0

                        total_accuracy_at1 += item_accuracy_at1
                        total_accuracy_at16 += item_accuracy_at16
                        total_major_accuracy_at16 += major_accuracy_at16
                        total_items += 1

                file_pass_at1 = total_accuracy_at1 / total_items if total_items else 0.0
                file_pass_at16 = total_accuracy_at16 / total_items if total_items else 0.0
                file_major_at16 = total_major_accuracy_at16 / total_items if total_items else 0.0

                print(f"  {file}: pass@1/pass@16/major@16 = {file_pass_at1:.4f}/{file_pass_at16:.4f}/{file_major_at16:.4f}", file=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accuracy metrics for model outputs')
    parser.add_argument('model_path', type=str, help='Path to the model')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Name of evaluation results directory')
    args = parser.parse_args()

    model_base_dir = (os.path.abspath(args.model_path))

    eval_results_dir = os.path.join(model_base_dir, args.output_dir)
    
    # 确保输出目录存在
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # 设置结果文件路径
    output_file_path = os.path.join(eval_results_dir, "cal_all.txt")
    
    print(f"Processing model outputs from: {model_base_dir}")
    print(f"\033[92mResults will be saved to: {output_file_path}\033[0m")
    
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        print(f"Processing model outputs from: {eval_results_dir}", file=output_file)
        compute_pass_at1(model_base_dir, output_file) 
