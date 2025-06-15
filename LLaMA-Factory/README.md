## Getting Started

### Installation

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Training with QFFT

To train a model using the QFFT method:

```bash
llamafactory-cli train examples/train_qfft/train_s1_qfft.yaml
```

```bash
llamafactory-cli train examples/train_qfft/train_limo_qfft.yaml
```

## Our Modifications

This codebase is primarily based on LLaMA-Factory. Our main changes are in the template. For details, please refer to `/src/llamafactory/data/template.py`, specifically at line 1569.


## Acknowledgement

This repo is built upon [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We thank the original authors for their excellent work.