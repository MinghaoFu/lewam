"""
Generate CLIP text embeddings for robomimic tasks and save back to robomimic_tasks.json.

Usage:
    python create_task_embeddings.py [--tasks_file robomimic_tasks.json]

Format follows Newt (Hansen et al., 2025):
    text = f"{embodiment}. {instruction}."
    embedding = CLIP_text_encoder(text).last_hidden_state.mean(dim=1)  # (512,)
"""

import argparse
import json
import os

import torch
from transformers import CLIPTokenizer, CLIPTextModel


def encode_tasks(tasks_file: str):
    with open(tasks_file, "r") as f:
        tasks = json.load(f)

    print(f"Loading CLIP tokenizer and text model (openai/clip-vit-base-patch32)...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_model.to(device)

    print(f"Encoding {len(tasks)} tasks...")
    for task_name, task_info in tasks.items():
        text = f'{task_info["embodiment"]}. {task_info["instruction"]}.'
        print(f"  [{task_name}] {text}")

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = text_model(**inputs)
            # pooler_output uses the [EOS] token position — CLIP's intended
            # text representation, avoids polluting with padding tokens
            embedding = output.pooler_output.squeeze(0)  # (512,)

        task_info["text_embedding"] = embedding.cpu().tolist()

    # overwrite the same file
    with open(tasks_file, "w") as f:
        json.dump(tasks, f, indent=2)

    dim = len(list(tasks.values())[0]["text_embedding"])
    print(f"\nDone. Saved embeddings of dim {dim} to {tasks_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks_file",
        type=str,
        default="robomimic_tasks.json",
        help="Path to tasks JSON file",
    )
    args = parser.parse_args()
    assert os.path.exists(args.tasks_file), f"File not found: {args.tasks_file}"
    encode_tasks(args.tasks_file)
