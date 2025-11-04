#!/bin/bash

export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1

output_dir=your_path_to_temp_sampling_output
tokenizer_model_path=your_path/Qwen2.5-7B-Instruct

model_names=(
    your_path/output/limo_qfft_7B
    your_path/output/s1_qfft_7B
)

data_names=(
    "aime25" "aime24" "gpqa" "math"
    # or add more data names here
)

# Start evaluation sampling
echo "Starting evaluation..."
for model_name in "${model_names[@]}"; do
    for data_name in "${data_names[@]}"; do

        CUDA_VISIBLE_DEVICES='0,1,2,3' \
        python eval.py \
        --model_name_or_path "$model_name" \
        --data_name "$data_name" \
        --prompt_type "qwen-instruct" \
        --temperature 0.6 \
        --start_idx 0 \
        --end_idx -1 \
        --n_sampling 16 \
        --k 16 \
        --split "test" \
        --max_tokens 32000 \
        --seed 42 \
        --top_p 1 \
        --surround_with_messages \
        --output_dir $output_dir 
        --num_gpus 4
    done
done
echo "Completed evaluation"

# Calculate accuracy metrics
echo "Calculating accuracy metrics..."
python analysis/cal_all_cli.py $output_dir

# Analyze outputs length
echo "Analyzing model outputs..."
python analysis/analysis_cli.py $output_dir --model_path $tokenizer_model_path