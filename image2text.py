import argparse
import torch
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--lora_path", type=str, required=False, help="Path to the fine-tuned LoRA model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument(
        "--question", type=str, default="What is in the image?", help="Question to ask about the image."
    )
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading VLChatProcessor from {args.model_path}...")
    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    print(f"[INFO] Loading base model from {args.model_path}...")
    base_model = (
        AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
        .to(torch.bfloat16)
        .to(device)
        .eval()
    )

    if args.lora_path is not None:
        print(f"[INFO] Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        model.eval()
    else:
        model = base_model

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{args.question}",
            "images": [args.image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    print("[INFO] Loading image and processing input...")
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(device)

    print("[INFO] Generating response...")
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print("\n=== Model Output ===\n")
    print(answer)


if __name__ == "__main__":
    main()
