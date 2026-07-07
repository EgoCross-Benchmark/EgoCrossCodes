# EgoCross Open-Source MLLM Closed-Set Evaluation

This repo contains utilities for benchmarking multimodal large language models (MLLMs) on the EgoCross QA tasks. It covers both locally hosted VLMs (Qwen, InternVL, VideoLLaMA) and the provided GPT/Gemini API evaluators.

## Directory Overview

- `eval_os_closedset.py`: closed-set evaluation for locally hosted VLMs.
- `eval_gpt_closedset.py`, `eval_gpt_openset.py`: GPT-4.1 evaluators for closed/open-set tasks.
- `eval_gemini_closedset.py`, `eval_gemini_openset.py`: Gemini 2.5 Pro evaluators for closed/open-set tasks.
- `preference.py`: central configuration for model identifiers and optional API keys. Update the values inside this file if you host checkpoints locally.
- `datasets/`: expected location for the EgoCross benchmark JSON file `egocross_testbed_imgs.json`.

## Environment Setup

The evaluators assume a CUDA-capable GPU (tested with CUDA 11.8). We recommend creating an isolated Conda environment:

```bash
conda create -n egocross python=3.12 \
    pytorch=2.2 torchvision torchaudio pytorch-cuda=11.8 ffmpeg \
    -c pytorch -c nvidia
conda activate egocross
pip install -r requirements.txt
```

`requirements.txt` (already in the repository root) contains only the libraries used directly by the `eval_*.py` scripts:

```
google-generativeai==0.8.5
openai==1.70.0
transformers==4.50.3
qwen-vl-utils==0.0.10
decord==0.6.0
Pillow==10.4.0
numpy==2.0.1
tqdm==4.67.1
safetensors==0.5.3
sentencepiece==0.2.0
huggingface-hub==0.30.1
```

After installation, update `preference.py` with the necessary API keys (OpenAI, Gemini, Qwen, etc.) before running any evaluator.

## Configuration

1. **Model identifiers**: `preference.py` contains default Hugging Face repo IDs. Replace them with local checkpoints if running fully offline.
2. **CUDA device**: select GPUs via the `CUDA_VISIBLE_DEVICES` environment variable as usual.
3. **Dataset**: place the EgoCross QA JSON file inside `datasets/`. To compute accuracy, the evaluators need the answer-bearing file `egocross_testbed_answers.json` (see [Answers & Offline Evaluation](#answers--offline-evaluation)) — download it from the [EgoCross HuggingFace dataset](https://huggingface.co/datasets/myuniverse/EgoCross) into this folder, or point `--dataset_path` to it directly. `egocross_testbed_imgs.json` (questions only) is what challenge participants used for generating predictions.

## Usage

Example command mirroring the previous workflow (Qwen2.5-VL closed-set evaluation on GPU 3):

```bash
CUDA_VISIBLE_DEVICES=3 python eval_os_closedset.py \
  --model_name qwen25vl \
  --dataset_path datasets/egocross_testbed_imgs.json \
  --output_dir results-egocross-testbed/closedset-Qwen25VL
```

Key flags:

- `--model_name`: one of `qwen2vl`, `qwen25vl`, `qwen25vl_3b`, `internvl25`, `internvl3`, `videollama3`.
- `--fps`: frame sampling rate (default `0.5`).
- `--max_inference_pixels`: uprate/lower to control preprocessing resolution.
- `--limit`: restrict evaluation to the first N samples (0 means full set).
- `--save_interval`: dump intermediate results every N samples to prevent data loss.

The script writes a timestamped JSON summary into `--output_dir`, containing both per-sample predictions and aggregated accuracy.

## Answers & Offline Evaluation

The 1st Cross-Domain EgoCross Challenge ([Source-Limited Track](https://www.codabench.org/competitions/11279/), [Open-Source Track](https://www.codabench.org/competitions/13868/); EgoVis Workshop @ CVPR 2026) concluded in May 2026. The **ground-truth answers** for all 957 testbed questions are now released, so evaluation can be run and scored fully offline without submitting to Codabench.

- Download `egocross_testbed/egocross_testbed_answers.json` from the [EgoCross HuggingFace dataset](https://huggingface.co/datasets/myuniverse/EgoCross) (gated access — request access on the dataset page first).
- It contains the same 957 questions as `egocross_testbed_imgs.json` (matched 1:1 by `id`), plus `correct_option_letter`, `answer_text`, `detailed_answer`, and a globally unique `question_id`.
- The evaluators in this repo (`eval_os_closedset.py`, `eval_gpt_*`, `eval_gemini_*`) read `correct_option_letter` / `answer_text` to compute accuracy, so point `--dataset_path` at the answers file:

```bash
CUDA_VISIBLE_DEVICES=3 python eval_os_closedset.py \
  --model_name qwen25vl \
  --dataset_path datasets/egocross_testbed_answers.json \
  --output_dir results-egocross-testbed/closedset-Qwen25VL
```

⚠️ **Do not train on the testbed.** The answers are released for evaluation and analysis only; please do not include the testbed questions or answers in any training corpus.

## Notes

- The official/reference evaluation scripts can be found at [EgoCrossCodes](https://github.com/EgoCross-Benchmark/EgoCrossCodes).
- InternVL loading requires `qwen_vl_utils.process_vision_info` and the model-specific `chat` API; ensure the respective repositories are installed.
- The pipeline assumes CUDA execution (`tensor.cuda()`). For CPU-only or other accelerator setups, additional adjustments are required.
- API-based evaluators (`eval_gpt_*`, `eval_gemini_*`) invoke external services; populate the relevant API keys in `preference.py` before running them.
