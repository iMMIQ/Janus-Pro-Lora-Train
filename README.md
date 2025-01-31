# Multi-Modal Fine-Tuning and Inference with Janus-Pro

This repository provides scripts to fine-tune and infer a multi-modal model based on Janus-Pro. The project leverages LoRA for parameter-efficient fine-tuning and supports image-text interactions.

## Environment Setup

Before running the scripts, prepare your environment:

```bash
pip install -r requirements.txt

git lfs install
git clone https://huggingface.co/deepseek-ai/Janus-Pro-1B

git clone https://github.com/deepseek-ai/Janus.git
mv Janus/janus janus
```

## Training: `train_janus_pro_lora.py`

### Description
This script fine-tunes the Janus-Pro model using LoRA on a dataset of image-text pairs.

### Usage

```bash
python train_janus_pro_lora.py --data_dir dataset/images \
    --pretrained_model Janus-Pro-1B \
    --output_dir ./janus_lora_output \
    --batch_size 2 \
    --max_epochs 15 \
    --lr 5e-5 \
    --seed 42
```

### Arguments
- `--data_dir`: Path to the directory containing images and text files.
- `--pretrained_model`: Name or path of the pre-trained Janus-Pro model.
- `--output_dir`: Directory to save the fine-tuned model.
- `--batch_size`: Number of samples per batch.
- `--max_epochs`: Number of training epochs.
- `--lr`: Learning rate.
- `--seed`: Random seed for reproducibility.

### Output
The fine-tuned model will be saved in `output_dir`, including the LoRA adapter and updated processor.

---

## Inference: `image2text.py`

### Description
This script performs inference using a fine-tuned Janus-Pro model with an optional LoRA adapter.

### Usage

```bash
python image2text.py --model_path ./janus_lora_output \
    --image_path sample.jpg \
    --question "What is in the image?" \
    --max_new_tokens 512
```

### Arguments
- `--model_path`: Path to the base Janus-Pro model.
- `--lora_path`: (Optional) Path to the fine-tuned LoRA adapter.
- `--image_path`: Path to the input image.
- `--question`: Text query to ask about the image.
- `--max_new_tokens`: Maximum number of generated tokens.

### Output
The script prints the generated text response to the console.

---

For further details, refer to the source code in `train_janus_pro_lora.py` and `image2text.py`.
