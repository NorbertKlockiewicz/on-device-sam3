"""Real weights: does SAM 3 at 672px find the same instances as at 1008px? Prints counts, scores, mask IoU."""
import sys, time
import torch, numpy as np
from PIL import Image
from transformers import Sam3Model, Sam3Processor, Sam3Config

CASES = [("coco_cats.jpg", ["cat", "remote control", "couch"]), ("coco_kitchen.jpg", ["person", "bowl", "bottle", "oven"])]

def run(model, proc, img, prompt, size):
    inputs = proc(images=img, text=prompt, return_tensors="pt", size={"height": size, "width": size})
    with torch.no_grad():
        t0 = time.time(); out = model(**inputs); ms = (time.time() - t0) * 1000
    res = proc.post_process_instance_segmentation(out, threshold=0.5, mask_threshold=0.5, target_sizes=[img.size[::-1]])[0]
    return res, ms

def iou(a, b):
    a, b = a.bool(), b.bool()
    return (a & b).sum().item() / max(1, (a | b).sum().item())

proc = Sam3Processor.from_pretrained("facebook/sam3")
models = {}
for size in (1008, 672):
    cfg = Sam3Config.from_pretrained("facebook/sam3")
    cfg.vision_config.image_size = size; cfg.vision_config.backbone_config.image_size = size
    models[size] = Sam3Model.from_pretrained("facebook/sam3", config=cfg).eval()

for fname, prompts in CASES:
    img = Image.open(fname).convert("RGB")
    for p in prompts:
        r_hi, ms_hi = run(models[1008], proc, img, p, 1008)
        r_lo, ms_lo = run(models[672], proc, img, p, 672)
        # Greedy match each 1008 instance to its best 672 mask.
        ious = []
        for m in r_hi["masks"]:
            ious.append(max([iou(m, n) for n in r_lo["masks"]], default=0.0))
        print(f"{fname:16s} '{p}': 1008 -> {len(r_hi['scores'])} inst {[round(s,2) for s in r_hi['scores'].tolist()]} ({ms_hi:.0f} ms) | "
              f"672 -> {len(r_lo['scores'])} inst {[round(s,2) for s in r_lo['scores'].tolist()]} ({ms_lo:.0f} ms) | best IoU per 1008 inst {[round(x,2) for x in ious]}", flush=True)
