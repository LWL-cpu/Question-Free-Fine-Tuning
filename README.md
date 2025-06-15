# QFFT, Question-Free Fine-Tuning for Adaptive Reasoning

<div align="center">
  <h3>QFFT</h3>
</div>

<p align="center">
📃 <a href="assets/paper.pdf" target="assets/paper.pdf">Paper</a> ｜ 🤗 <a href="https://huggingface.co/lwl-uestc/QFFT-S1-7B" target="_blank">QFFT-7B</a> ｜ 🤗 <a href="https://huggingface.co/lwl-uestc/QFFT-S1-32B" target="_blank">QFFT-32B</a> ｜ 📚 <a href="https://huggingface.co/datasets/lwl-uestc/S1_QFFT">QFFT Datasets</a>
</p>

---

## ⚡ Introduction

Welcome to the official repository for **QFFT, Question-Free Fine-Tuning for Adaptive Reasoning**!


QFFT introduces a novel and efficient fine-tuning method designed to empower large language models with **adaptive reasoning ability**. Instead of training models on (Question, Reasoning) pairs like traditional Supervised Fine-Tuning (SFT), **QFFT discards the question input and learns solely from the reasoning response**—especially Long CoT outputs.

QFFT enables models to:

- **Preserve Short CoT** for simple tasks (efficiency)
- **Trigger Long CoT** only when needed (effectiveness)
- **Reduce overthinking** by minimizing unnecessary reasoning
- **Improve robustness** in noisy, low-resource, and out-of-domain scenarios

We open-sourced our models, data, and code here.

---

## 💭 Environment

### Training Environment (LLaMA-Factory)

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Evaluation Environment (VLLM)

```bash
pip install vllm bitsandbytes flashinfer-python==0.2.2.post1
pip install latex2sympy2 word2number
```

---

## 💻 Model

| Model Name           | Base LLM              | Link                                                                   |
|----------------------|-----------------------|------------------------------------------------------------------------|
| **QFFT-S1-7B**       | Qwen2.5-7B-Instruct    | [HF Link](https://huggingface.co/lwl-uestc/QFFT-S1-7B)                |
| **QFFT-S1-32B**      | Qwen2.5-32B-Instruct   | [HF Link](https://huggingface.co/lwl-uestc/QFFT-S1-32B)               |
| **QFFT-LIMO-7B**     | Qwen2.5-7B-Instruct    | [HF Link](https://huggingface.co/lwl-uestc/QFFT-LIMO-7B)              |
| **QFFT-LIMO-32B**    | Qwen2.5-32B-Instruct   | [HF Link](https://huggingface.co/lwl-uestc/QFFT-LIMO-32B)             |

---

## 📚 Datasets

QFFT uses distilled responses from strong Long CoT models (e.g., DeepSeek-R1). During QFFT, the input questions are removed entirely.

| Dataset             | Size   | Description                        |
|---------------------|--------|------------------------------------|
| S1.1                | 1k     | High-quality diverse reasoning     |
| LIMO                | 871    | Long CoT distilled samples         |

---

## 🛠️ Training

### Getting Started

To train a model using QFFT, you can use `llamafactory-cli` and the provided YAML configs:

```bash
llamafactory-cli train examples/train_qfft/train_s1_qfft.yaml
llamafactory-cli train examples/train_qfft/train_limo_qfft.yaml
```

### Our Modifications

This codebase is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).  
Our key modification lies in the template system. We implement a new QFFT template in:

```
/src/llamafactory/data/template.py
```

For details, please refer line **1569**.

---

## 🧪 Evaluation

You can evaluate QFFT models using standard benchmark suites (GSM8K, MATH, AIME) with tools like `vllm` or `Sglang`.  
We also propose a novel metric **RAK** (Reasoning Adaptability Kappa) to evaluate reasoning strategy alignment with task difficulty.

---

## 📊 Results

Here are the main results comparing SFT and QFFT on 3 mathematical reasoning benchmarks:

### 📌 7B Models (Qwen2.5-7B-Instruct)

| Dataset | Method | GSM8K Acc | GSM8K Tokens | MATH Acc | MATH Tokens | AIME25 Acc | AIME25 Tokens | Avg Acc | Avg Tokens |
|---------|--------|-----------|--------------|----------|-------------|------------|----------------|---------|-------------|
| S1.1    | SFT    | 90.6      | 1.7K         | 80.8     | 5.3K        | 18.2       | 17.7K          | 63.2    | 8.2K        |
|         | QFFT   | 91.0      | 0.4K         | 80.2     | 2.8K        | 17.2       | 12.8K          | 62.8    | 5.3K        |
|         | Δ      | +0.4      | -76.5%       | -0.6     | -47.2%      | -1.0       | -27.7%         | -0.4    | -50.5%      |

| Dataset | Method | GSM8K Acc | GSM8K Tokens | MATH Acc | MATH Tokens | AIME25 Acc | AIME25 Tokens | Avg Acc | Avg Tokens |
|---------|--------|-----------|--------------|----------|-------------|------------|----------------|---------|-------------|
| LIMO    | SFT    | 88.2      | 1.8K         | 80.4     | 5.8K        | 16.8       | 17.1K          | 61.8    | 8.2K        |
|         | QFFT   | 88.0      | 0.7K         | 80.6     | 4.1K        | 17.2       | 15.6K          | 61.9    | 6.8K        |
|         | Δ      | -0.2      | -61.1%       | +0.2     | -29.3%      | +0.4       | -8.8%          | +0.1    | -33.1%      |

### 📌 32B Models (Qwen2.5-32B-Instruct)

| Dataset | Method | GSM8K Acc | GSM8K Tokens | MATH Acc | MATH Tokens | AIME25 Acc | AIME25 Tokens | Avg Acc | Avg Tokens |
|---------|--------|-----------|--------------|----------|-------------|------------|----------------|---------|-------------|
| S1.1    | SFT    | 92.8      | 2.1K         | 93.1     | 4.1K        | 48.6       | 16.2K          | 78.2    | 7.5K        |
|         | QFFT   | 93.6      | 0.6K         | 92.2     | 2.4K        | 46.8       | 12.9K          | 77.5    | 5.3K        |
|         | Δ      | +0.8      | -71.4%       | -0.9     | -41.5%      | -1.8       | -20.4%         | -0.6    | -44.4%      |

| Dataset | Method | GSM8K Acc | GSM8K Tokens | MATH Acc | MATH Tokens | AIME25 Acc | AIME25 Tokens | Avg Acc | Avg Tokens |
|---------|--------|-----------|--------------|----------|-------------|------------|----------------|---------|-------------|
| LIMO    | SFT    | 91.2      | 1.9K         | 93.0     | 3.9K        | 45.8       | 13.2K          | 76.6    | 6.3K        |
|         | QFFT   | 92.6      | 0.8K         | 92.6     | 2.9K        | 45.0       | 12.5K          | 76.7    | 5.4K        |
|         | Δ      | +1.4      | -57.9%       | -0.4     | -25.6%      | -0.8       | -5.3%          | +0.1    | -29.6%      |



---

## 📖 Citation

```
@misc{liu2025qfft,
  title={QFFT, Question-Free Fine-Tuning for Adaptive Reasoning},
  author={Wanlong Liu and Junxiao Xu and Fei Yu and Yukang Lin and Ke Ji and Wenyu Chen and Yan Xu and Yasheng Wang and Lifeng Shang and Benyou Wang},
  year={2025},
  eprint={TBD},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
