import json
import re
from collections import defaultdict
import numpy as np

def load_math_test_levels(file_path):
    """Load question-to-level mapping from math test JSONL."""
    question_to_level = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            question = item.get('problem', '')
            level = item.get('level', None)
            if question and level is not None:
                question_to_level[question] = int(level)
    return question_to_level

def contains_wait(response):
    """Check if response contains the word 'wait' (case-insensitive)."""
    return bool(re.search(r'\bwait\b', response.lower(), re.IGNORECASE))

def extract_question(problem):
    """Extract question text between <|im_start|>user\n and <|im_end|>\n, or return problem if no markers."""
    if "<|im_start|>" in problem:
        pattern = r'<\|im_start\|>user\n(.*?)<\|im_end\|>\n'
        match = re.search(pattern, problem, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ''
    else:
        return problem

def process_jsonl(input_file, math_test_file):
    # Load level information from math test file
    question_to_level = load_math_test_levels(math_test_file)
    
    # Initialize lists to store per-item statistics
    total_items = 0
    item_wait_proportions = []
    item_wait_accuracies = []
    item_no_wait_proportions = []
    item_no_wait_accuracies = []
    
    # Lists to store items with all/no wait
    all_wait_items = []
    all_no_wait_items = []
    
    # Per-level statistics (store per-item stats for averaging)
    level_stats = defaultdict(lambda: {
        'wait_proportions': [],
        'wait_accuracies': [],
        'no_wait_proportions': [],
        'no_wait_accuracies': [],
        'total_items': 0,
        'total_responses': 0,
        'wait_lengths': [],
        'no_wait_lengths': []
    })
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            problem = item.get('question', '')
            # Extract question from problem field
            question = extract_question(problem)
            responses = item.get('generated_responses', [])
            correctness = item.get('answers_correctness', [])
            
            if not responses or not correctness or len(responses) != len(correctness):
                continue
            
            total_items += 1
            item_response_count = len(responses)
            
            # Get level for this question
            level = question_to_level.get(question, None)
            
            # Process each response
            item_wait_count = 0
            item_wait_correct = 0
            item_no_wait_count = 0
            item_no_wait_correct = 0
            
            for response, is_correct in zip(responses, correctness):
                has_wait = contains_wait(response)
                word_count = len(response.split())
                
                if has_wait:
                    item_wait_count += 1
                    if is_correct:
                        item_wait_correct += 1
                    if level is not None:
                        level_stats[level]['wait_lengths'].append(word_count)
                else:
                    item_no_wait_count += 1
                    if is_correct:
                        item_no_wait_correct += 1
                    if level is not None:
                        level_stats[level]['no_wait_lengths'].append(word_count)
            
            # Check if all responses have/no wait
            if item_wait_count == item_response_count and item_response_count > 0:
                all_wait_items.append({'question': question, 'level': level})
            elif item_no_wait_count == item_response_count and item_response_count > 0:
                all_no_wait_items.append({'question': question, 'level': level})
            
            # Calculate item-level statistics
            wait_proportion = item_wait_count / item_response_count if item_response_count > 0 else 0
            wait_accuracy = item_wait_correct / item_wait_count if item_wait_count > 0 else 0
            no_wait_proportion = item_no_wait_count / item_response_count if item_response_count > 0 else 0
            no_wait_accuracy = item_no_wait_correct / item_no_wait_count if item_no_wait_count > 0 else 0
            
            # Store item-level statistics
            item_wait_proportions.append(wait_proportion)
            item_wait_accuracies.append(wait_accuracy)
            item_no_wait_proportions.append(no_wait_proportion)
            item_no_wait_accuracies.append(no_wait_accuracy)
            
            # Update level stats
            if level is not None:
                level_stats[level]['wait_proportions'].append(wait_proportion)
                level_stats[level]['wait_accuracies'].append(wait_accuracy)
                level_stats[level]['no_wait_proportions'].append(no_wait_proportion)
                level_stats[level]['no_wait_accuracies'].append(no_wait_accuracy)
                level_stats[level]['total_items'] += 1
                level_stats[level]['total_responses'] += item_response_count
            
            # Print item-level stats
            print(f"\nItem {total_items} (Question: {question[:50]}...):")
            print(f"  Responses with 'wait': {item_wait_count}/{item_response_count} ({wait_proportion:.2%})")
            print(f"  Accuracy with 'wait': {wait_accuracy:.2%}")
            print(f"  Responses without 'wait': {item_no_wait_count}/{item_response_count} ({no_wait_proportion:.2%})")
            print(f"  Accuracy without 'wait': {no_wait_accuracy:.2%}")
    
    # Print items with all responses containing wait
    print("\n=== Items with All Responses Containing 'wait' ===")
    if all_wait_items:
        for idx, item in enumerate(all_wait_items, 1):
            level_str = item['level'] if item['level'] is not None else 'Unknown'
            print(f"Item {idx}:")
            print(f"  Question: {item['question'][:100]}...")
            print(f"  Level: {level_str}")
    else:
        print("No items found where all responses contain 'wait'.")
    
    # Print items with all responses not containing wait
    print("\n=== Items with All Responses Not Containing 'wait' ===")
    if all_no_wait_items:
        for idx, item in enumerate(all_no_wait_items, 1):
            level_str = item['level'] if item['level'] is not None else 'Unknown'
            print(f"Item {idx}:")
            print(f"  Question: {item['question'][:100]}...")
            print(f"  Level: {level_str}")
    else:
        print("No items found where all responses do not contain 'wait'.")
    
    # Overall statistics (average of per-item stats)
    avg_wait_proportion = np.mean(item_wait_proportions) if item_wait_proportions else 0
    avg_wait_accuracy = np.mean(item_wait_accuracies) if item_wait_accuracies else 0
    avg_no_wait_proportion = np.mean(item_no_wait_proportions) if item_no_wait_proportions else 0
    avg_no_wait_accuracy = np.mean(item_no_wait_accuracies) if item_no_wait_accuracies else 0
    
    print("\n=== Overall Statistics ===")
    print(f"Total items processed: {total_items}")
    print(f"Average proportion with 'wait': {avg_wait_proportion:.2%}")
    print(f"Average accuracy with 'wait': {avg_wait_accuracy:.2%}")
    print(f"Average proportion without 'wait': {avg_no_wait_proportion:.2%}")
    print(f"Average accuracy with 'wait': {avg_no_wait_accuracy:.2%}")
    
    # Per-level statistics (average of per-item stats)
    print("\n=== Per-Level Statistics ===")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        total_items_level = stats['total_items']
        if total_items_level == 0:
            continue
        avg_wait_prop = np.mean(stats['wait_proportions']) if stats['wait_proportions'] else 0
        avg_wait_acc = np.mean(stats['wait_accuracies']) if stats['wait_accuracies'] else 0
        avg_no_wait_prop = np.mean(stats['no_wait_proportions']) if stats['no_wait_proportions'] else 0
        avg_no_wait_acc = np.mean(stats['no_wait_accuracies']) if stats['no_wait_accuracies'] else 0
        avg_wait_length = np.mean(stats['wait_lengths']) if stats['wait_lengths'] else 0
        avg_no_wait_length = np.mean(stats['no_wait_lengths']) if stats['no_wait_lengths'] else 0
        
        print(f"\nLevel {level}:")
        print(f"  Total items: {total_items_level}")
        print(f"  Total responses: {stats['total_responses']}")
        print(f"  Average proportion with 'wait': {avg_wait_prop:.2%}")
        print(f"  Average accuracy with 'wait': {avg_wait_acc:.2%}")
        print(f"  Average response length with 'wait': {avg_wait_length:.2f} words")
        print(f"  Average proportion without 'wait': {avg_no_wait_prop:.2%}")
        print(f"  Average accuracy without 'wait': {avg_no_wait_acc:.2%}")
        print(f"  Average response length without 'wait': {avg_no_wait_length:.2f} words")

if __name__ == '__main__':
    input_file = '/wangbenyou/wanlong/SFT/eval/outputs/output/ours_0417/2k_instruct_noq/math/test_qwen-instruct_t0.6_k16_s0_e500.jsonl'
    math_test_file = '/wangbenyou/wanlong/SFT/eval/data/math/test.jsonl'
    process_jsonl(input_file, math_test_file)