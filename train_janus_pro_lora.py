import os
import argparse
import random
import logging
import torch
import numpy as np

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from janus.models import MultiModalityCausalLM
from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/images")
    parser.add_argument("--pretrained_model", type=str, default="Janus-Pro-1B")
    parser.add_argument("--output_dir", type=str, default="./janus_lora_output")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_image_text_pairs(data_dir):
    pairs = []
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_files, txt_files = {}, {}

    for root, _, files in os.walk(data_dir):
        for fname in files:
            base, ext = os.path.splitext(fname)
            if ext.lower() in img_exts:
                img_files[base] = os.path.join(root, fname)
            elif ext.lower() == ".txt":
                txt_files[base] = os.path.join(root, fname)

    for k, v in txt_files.items():
        if k in img_files:
            with open(v, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            pairs.append((img_files[k], txt))

    if not pairs:
        raise ValueError(f"No matched image-text pairs in {data_dir}.")
    return pairs


class MultiModalTrainDataset(Dataset):
    def __init__(self, pairs):
        self.data = pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, txt = self.data[idx]
        return {"image_path": img_path, "text": txt}


def conversation_template(image_path, user_text="What is in the image?", assistant_text=""):
    return [
        {"role": "<|User|>", "content": f"<image_placeholder>\n{user_text}", "images": [image_path]},
        {"role": "<|Assistant|>", "content": assistant_text},
    ]


def collate_fn_for_vlchat(batch, processor: VLChatProcessor):
    conversations = []
    for item in batch:
        conversations.extend(conversation_template(image_path=item["image_path"], assistant_text=item["text"]))

    pil_images_list = load_pil_images(conversations)
    encoded = processor(
        conversations=conversations,
        images=pil_images_list,
        return_tensors="pt",
        force_batchify=True,
    )
    encoded["labels"] = encoded["input_ids"].clone()
    return dict(encoded)


def new_forward(
    self,
    input_ids=None,
    attention_mask=None,
    pixel_values=None,
    images_seq_mask=None,
    images_emb_mask=None,
    labels=None,
    **kwargs,
):
    inputs_embeds = None
    if pixel_values is not None:
        inputs_embeds = self.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )

    if inputs_embeds is not None:
        kwargs.pop("inputs_embeds", None)
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    else:
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
    return outputs


MultiModalityCausalLM.forward = new_forward


def main():
    args = parse_args()
    set_seed(args.seed)

    pairs = find_image_text_pairs(args.data_dir)
    logging.info(f"Found {len(pairs)} samples in {args.data_dir} for multi-modal training.")
    dataset = MultiModalTrainDataset(pairs)

    logging.debug("Checking first few samples in dataset:")
    for i in range(min(5, len(dataset))):
        logging.debug(f"  dataset[{i}]: {dataset[i]}")

    logging.info(f"Loading pre-trained model from: {args.pretrained_model}")
    processor = VLChatProcessor.from_pretrained(args.pretrained_model)
    model = MultiModalityCausalLM.from_pretrained(args.pretrained_model, torch_dtype=torch.float16, device_map="auto")
    model.train()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda features: collate_fn_for_vlchat(features, processor),
    )

    logging.debug("Start training ...")
    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logging.info(f"Fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
