"""End-to-end: text.pte + 4 vision chunks + head.pte on real photos vs PyTorch eager, compared after post-processing."""
import torch, numpy as np
from PIL import Image
from transformers import Sam3Model, Sam3Processor, Sam3Config
from executorch.runtime import Runtime

S = 672
rt = Runtime.get()
def load(p): return rt.load_program(f"pte_real/{p}").load_method("forward")
text_m = load(f"sam3_text_{S}_coreml.pte")
vis_m = [load(f"sam3_vision{i}_{S}_coreml.pte") for i in range(4)]
head_m = load(f"sam3_head_{S}_coreml.pte")

proc = Sam3Processor.from_pretrained("facebook/sam3")
cfg = Sam3Config.from_pretrained("facebook/sam3")
cfg.vision_config.image_size = S; cfg.vision_config.backbone_config.image_size = S
model = Sam3Model.from_pretrained("facebook/sam3", config=cfg).eval()

class Out:  # minimal stand-in for Sam3ImageSegmentationOutput for the post-processor
    def __init__(self, masks, boxes, logits, presence):
        self.pred_masks, self.pred_boxes, self.pred_logits, self.presence_logits = masks, boxes, logits, presence
        self.semantic_seg = None
    def __getitem__(self, k): return getattr(self, k)
    def get(self, k, d=None): return getattr(self, k, d)

def iou(a, b):
    a, b = a.bool(), b.bool(); return (a & b).sum().item() / max(1, (a | b).sum().item())

for fname, prompts in [("coco_cats.jpg", ["cat", "remote control"]), ("coco_kitchen.jpg", ["person", "bowl", "oven"])]:
    img = Image.open(fname).convert("RGB")
    for p in prompts:
        inp = proc(images=img, text=p, return_tensors="pt", size={"height": S, "width": S})
        ids = inp["input_ids"]; am = inp["attention_mask"]
        assert ids.shape == (1, 32), ids.shape
        with torch.no_grad():
            ref = model(**inp)
        tf = text_m.execute([ids, am])[0]
        h = inp["pixel_values"]
        for i, m in enumerate(vis_m):
            o = m.execute([h])
            h = o[0] if i < 3 else o
        masks, boxes, logits, presence = head_m.execute([*h, tf, am])
        # Head returns boxes as cxcywh; the model output carries xyxy.
        cx, cy, w, hh = boxes.unbind(-1)
        boxes_xyxy = torch.stack([cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2], -1)
        ours = Out(masks, boxes_xyxy, logits, presence)
        r_ref = proc.post_process_instance_segmentation(ref, threshold=0.5, mask_threshold=0.5, target_sizes=[img.size[::-1]])[0]
        r_our = proc.post_process_instance_segmentation(ours, threshold=0.5, mask_threshold=0.5, target_sizes=[img.size[::-1]])[0]
        ious = [round(max([iou(m, n) for n in r_our["masks"]], default=0.0), 3) for m in r_ref["masks"]]
        print(f"{fname:16s} '{p}': torch {len(r_ref['scores'])} inst {[round(s,2) for s in r_ref['scores'].tolist()]} | "
              f"executorch {len(r_our['scores'])} inst {[round(s,2) for s in r_our['scores'].tolist()]} | mask IoU {ious} | "
              f"raw: logits diff {(logits-ref.pred_logits).abs().max():.2f}, mask-logit diff {(masks-ref.pred_masks).abs().max():.1f} (range {ref.pred_masks.abs().max():.0f})", flush=True)
