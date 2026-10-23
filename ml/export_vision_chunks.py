"""Export the SAM 3 ViT as 4 Core ML programs of 8 layers each (one program is too big for the ANE compiler).

  chunk0: pixel_values [1,3,S,S] -> embeddings + layernorm + layers 0..7 -> h [1,g,g,1024]
  chunk1: h -> layers 8..15 -> h
  chunk2: h -> layers 16..23 -> h
  chunk3: h -> layers 24..31 -> neck -> fpn0, fpn1, fpn2 (contiguous)
"""
import argparse, os, time
import torch
from torch import nn
from transformers import Sam3Config, Sam3Model
from export_sam3 import lower, run_pte


class Chunk(nn.Module):
    def __init__(self, vision, a, b, first, last, g):
        super().__init__()
        vit = vision.backbone
        self.embeddings = vit.embeddings if first else None
        self.layer_norm = vit.layer_norm if first else None
        self.layers = nn.ModuleList(vit.layers[a:b])
        self.neck = vision.neck if last else None
        self.g = g

    def forward(self, x):
        if self.embeddings is not None:
            h = self.embeddings(x).view(1, self.g, self.g, -1)
            h = self.layer_norm(h)
        else:
            h = x
        for l in self.layers:
            h = l(h)
        if self.neck is None:
            return h
        fpn, _ = self.neck(h.permute(0, 3, 1, 2))
        return tuple(f.contiguous() for f in fpn[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--size", type=int, default=672)
    ap.add_argument("--chunks", default="0,1,2,3")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.random:
        cfg = Sam3Config(); cfg.vision_config.image_size = a.size; cfg.vision_config.backbone_config.image_size = a.size
        model = Sam3Model(cfg).eval()
    else:
        cfg = Sam3Config.from_pretrained("facebook/sam3")
        cfg.vision_config.image_size = a.size; cfg.vision_config.backbone_config.image_size = a.size
        model = Sam3Model.from_pretrained("facebook/sam3", config=cfg).eval()
    vision = model.vision_encoder
    g = a.size // 14
    px = torch.randn(1, 3, a.size, a.size)
    with torch.no_grad():
        ref = tuple(f.contiguous() for f in vision(px).fpn_hidden_states[:-1])
    x, total = px, 0.0
    want = {int(c) for c in a.chunks.split(",")}
    for i, (lo, hi) in enumerate([(0, 8), (8, 16), (16, 24), (24, 32)]):
        m = Chunk(vision, lo, hi, i == 0, i == 3, g).eval()
        with torch.no_grad():
            y = m(x)
        if i not in want:
            x = y
            continue
        ep = torch.export.export(m, (x,), strict=False)
        path = f"sam3_vision{i}_{a.size}_coreml{a.suffix}{'_random' if a.random else ''}.pte"
        open(path, "wb").write(lower(ep, "coreml").buffer)
        out, ms = run_pte(path, (x,), n=3)
        total += ms
        print(f"chunk{i} layers {lo}:{hi} -> {path} {os.path.getsize(path)/1e6:.0f} MB, {ms:.0f} ms on Mac", flush=True)
        x = y
    diff = max((o - r).abs().max().item() for o, r in zip(out, ref))
    print(f"total {total:.0f} ms; last-chunk vs full eager vision (chained in eager) max diff {diff:.1e}")


if __name__ == "__main__":
    main()
