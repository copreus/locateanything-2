"""
RunPod Serverless handler for nvidia/LocateAnything-3B
(vision-language grounding / object localization model)

Model card: https://huggingface.co/nvidia/LocateAnything-3B

This mirrors the "Worker (recommended)" example from the model card as
closely as possible, adapted to:
  - load weights from RunPod's cached-model snapshot instead of
    downloading them from Hugging Face on every cold start
  - accept plain-JSON input (image as a URL or base64 string) over
    RunPod's Serverless API instead of local file paths
"""

import base64
import io
import os
import re
import traceback

import requests
import runpod
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_ID = "nvidia/LocateAnything-3B"

# --- Use RunPod's cached-model feature instead of downloading at cold start ---
# Set the endpoint's "Model" field to nvidia/LocateAnything-3B in the RunPod
# console. RunPod then pre-loads the repo onto the worker's host before your
# container even starts. These two env vars stop transformers/huggingface_hub
# from trying to reach the internet, so we always read the pre-cached copy.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"


def resolve_snapshot_path(model_id: str) -> str:
    """Find the local folder RunPod downloaded the cached model into."""
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_dir):
        versions = [
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(
        f"Cached model not found for '{model_id}'. Did you set the "
        f"endpoint's 'Model' field to {model_id!r}?"
    )


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_PATH = resolve_snapshot_path(MODEL_ID)

# Everything from here to the end of this block runs ONCE when the worker
# boots, not on every request -- this is what makes warm requests fast.
print(f"[worker] loading {MODEL_ID} from {MODEL_PATH} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True
)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True
)
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    local_files_only=True,
).to(DEVICE).eval()
print("[worker] model loaded, ready for requests.")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def load_image(job_input: dict) -> Image.Image:
    if job_input.get("image_base64"):
        data = job_input["image_base64"]
        if data.strip().startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

    if job_input.get("image_url"):
        resp = requests.get(job_input["image_url"], timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    raise ValueError("Provide either 'image_url' or 'image_base64' in the input.")


# Prompt templates copied from the "Supported Tasks & Prompt Templates"
# table on the model card, so behavior matches NVIDIA's documented API.
def build_prompt(job_input: dict) -> str:
    task = job_input.get("task", "prompt")

    if task == "prompt":
        prompt = job_input.get("prompt")
        if not prompt:
            raise ValueError("task='prompt' requires a 'prompt' string.")
        return prompt

    if task == "detect":
        categories = job_input.get("categories") or []
        if not categories:
            raise ValueError("task='detect' requires a non-empty 'categories' list.")
        return (
            "Locate all the instances that matches the following description: "
            f"{'</c>'.join(categories)}."
        )

    phrase = job_input.get("phrase", "")

    if task == "ground_single":
        return f"Locate a single instance that matches the following description: {phrase}."
    if task == "ground_multi":
        return f"Locate all the instances that match the following description: {phrase}."
    if task == "ground_text":
        return f"Please locate the text referred as {phrase}."
    if task == "detect_text":
        return "Detect all the text in box format."
    if task == "point":
        return f"Point to: {phrase}."
    if task == "ground_gui":
        if job_input.get("output_type") == "point":
            return f"Point to: {phrase}."
        return f"Locate the region that matches the following description: {phrase}."

    raise ValueError(
        f"Unknown task '{task}'. Use one of: prompt, detect, ground_single, "
        "ground_multi, ground_text, detect_text, point, ground_gui."
    )


# ---------------------------------------------------------------------------
# Inference -- adapted from LocateAnythingWorker.predict() on the model card
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    image: Image.Image,
    question: str,
    generation_mode: str = "hybrid",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # NOTE: the model card's own example calls `processor.py_apply_chat_template`,
    # not the usual `apply_chat_template`. This looks like it could be a typo in
    # NVIDIA's docs, but since this model ships custom processor code via
    # trust_remote_code, it's called exactly as documented. If this raises an
    # AttributeError during your first test, try `processor.apply_chat_template`.
    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = processor.process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, return_tensors="pt"
    ).to(DEVICE)

    pixel_values = inputs["pixel_values"].to(DTYPE)
    image_grid_hws = inputs.get("image_grid_hws", None)

    output = model.generate(
        pixel_values=pixel_values,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=image_grid_hws,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        generation_mode=generation_mode,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        verbose=False,
    )

    return output[0] if isinstance(output, tuple) else output


def parse_boxes(answer: str, width: int, height: int) -> list:
    """Model outputs coords normalized to [0, 1000]; scale to pixels."""
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        boxes.append({
            "x1": x1 / 1000 * width,
            "y1": y1 / 1000 * height,
            "x2": x2 / 1000 * width,
            "y2": y2 / 1000 * height,
        })
    return boxes


def parse_points(answer: str, width: int, height: int) -> list:
    points = []
    for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
        x, y = int(m.group(1)), int(m.group(2))
        points.append({"x": x / 1000 * width, "y": y / 1000 * height})
    return points


# ---------------------------------------------------------------------------
# RunPod entry point
# ---------------------------------------------------------------------------

def handler(job):
    job_input = job.get("input") or {}

    try:
        image = load_image(job_input)
        question = build_prompt(job_input)
        generation_mode = job_input.get("generation_mode", "hybrid")
        max_new_tokens = int(job_input.get("max_new_tokens", 2048))
        temperature = float(job_input.get("temperature", 0.7))

        answer = run_inference(
            image,
            question,
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        width, height = image.size

        return {
            "answer": answer,
            "boxes": parse_boxes(answer, width, height),
            "points": parse_points(answer, width, height),
            "image_size": {"width": width, "height": height},
        }

    except Exception as exc:
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})