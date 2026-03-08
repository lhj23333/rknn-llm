import torch
import os
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--path', type=str, required=True)
argparse.add_argument('--model_name', type=str, required=True,
                      choices=['qwen3-vl', 'internvl3.5'])
argparse.add_argument('--dataset_json', type=str, default='data/datasets.json')
argparse.add_argument('--output_json', type=str, default='data/inputs.json')
args = argparse.parse_args()

path = args.path
datasets = json.load(open(args.dataset_json, 'r'))
os.makedirs("data/inputs_embeds/", exist_ok=True)

if args.model_name == 'qwen3-vl':
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()

    print("model loaded")

    processor = AutoProcessor.from_pretrained(path)
    print("processor loaded")

    for data in datasets:
        image_name = data["image"].split(".")[0]
        imgp = os.path.join(data["image_path"], data["image"])
        image = Image.open(imgp).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": data["input"]},
                ],
            }
        ]

        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])

        pixel_values = inputs["pixel_values"].to(dtype=torch.float32)
        image_mask = inputs["input_ids"] == model.config.image_token_id

        image_embeds = model.visual(
            pixel_values,
            grid_thw=inputs["image_grid_thw"]
        )

        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]

        image_embeds = image_embeds.to(inputs_embeds.device)

        print("inputs_embeds shape:", inputs_embeds.shape)
        print("image_embeds shape:", image_embeds.shape)
        print("image token count:", image_mask.sum().item())

        inputs_embeds[image_mask] = image_embeds

        np.save(
            "data/inputs_embeds/{}".format(image_name),
            inputs_embeds.to(dtype=torch.float16).cpu().detach().numpy()
        )

elif args.model_name == 'internvl3.5':
    model = AutoModel.from_pretrained(
        path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    transform = build_transform(448)

    for data in datasets:
        image_name = data["image"].split(".")[0]
        imgp = os.path.join(data["image_path"], data["image"])
        image = Image.open(imgp).convert("RGB")

        pixel_values = transform(image).unsqueeze(0).to(model.device).to(torch.bfloat16)

        if hasattr(model, "extract_feature"):
            image_embeds = model.extract_feature(pixel_values)
        elif hasattr(model, "extract_vision_feature"):
            image_embeds = model.extract_vision_feature(pixel_values)
        else:
            raise Exception("no extract_feature / extract_vision_feature")

        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]

        if image_embeds.dim() == 3 and image_embeds.size(0) == 1:
            image_embeds = image_embeds[0]   # [1, N, C] -> [N, C]

        if image_embeds.dim() != 2:
            raise ValueError(f"Unexpected image_embeds shape: {image_embeds.shape}")

        num_image_token = image_embeds.shape[0]

        question = "<img>" + "".join(["<IMG_CONTEXT>" for _ in range(num_image_token)]) + "</img>" + data["input"]
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)

        if hasattr(model, "language_model") and hasattr(model.language_model, "get_input_embeddings"):
            inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
        elif hasattr(model, "get_input_embeddings"):
            inputs_embeds = model.get_input_embeddings()(input_ids)
        else:
            raise ValueError("Cannot locate text embedding layer")

        img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if img_context_id is None:
            raise ValueError("Cannot find <IMG_CONTEXT> token id")

        mask = (input_ids == img_context_id)

        print("image_embeds shape:", image_embeds.shape)
        print("num_image_token:", num_image_token)
        print("mask.sum():", mask.sum().item())
        print("inputs_embeds shape:", inputs_embeds.shape)

        if mask.sum().item() != num_image_token:
            raise ValueError(
                f"IMG_CONTEXT token count mismatch: mask.sum()={mask.sum().item()}, "
                f"num_image_token={num_image_token}"
            )

        inputs_embeds[mask] = image_embeds.to(inputs_embeds.dtype)

        print("final inputs_embeds:", inputs_embeds.shape)

        np.save(
            "data/inputs_embeds/{}".format(image_name),
            inputs_embeds.to(dtype=torch.float16).cpu().detach().numpy()
        )

with open(args.output_json, 'w') as json_file:
    json_file.write('[\n')
    first = True
    for data in tqdm(datasets):
        input_embed = np.load(os.path.join("data/inputs_embeds", data["image"].split(".")[0]+'.npy'))
        target = data["target"]
        input_dict = {
            "input_embed": input_embed.tolist(),
            "target": target
        }
        if not first:
            json_file.write(',\n')
        else:
            first = False
        json.dump(input_dict, json_file)
    json_file.write('\n]')

print("Done")
