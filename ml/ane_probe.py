"""Probe which ViT layer slices the ANE compiler accepts. Prints ms and whether ANE compiled."""
import sys, time, os, contextlib, io
import torch
from torch import nn
from transformers import Sam3Config, Sam3Model
from export_sam3 import lower, run_pte

size = int(sys.argv[1]); slices = [tuple(map(int, s.split(":"))) for s in sys.argv[2:]]
torch.manual_seed(0)
cfg = Sam3Config(); cfg.vision_config.image_size = size; cfg.vision_config.backbone_config.image_size = size
vit = Sam3Model(cfg).eval().vision_encoder.backbone
g = size // 14

class Slice(nn.Module):
    def __init__(self, a, b):
        super().__init__(); self.layers = nn.ModuleList(vit.layers[a:b])
    def forward(self, h):
        for l in self.layers: h = l(h)
        return h

x = torch.randn(1, g, g, 1024)
for a, b in slices:
    m = Slice(a, b).eval()
    with torch.no_grad(): ref = m(x)
    ep = torch.export.export(m, (x,), strict=False)
    path = f"probe_{size}_{a}_{b}.pte"
    open(path, "wb").write(lower(ep, "coreml").buffer)
    r, w = os.pipe(); saved = os.dup(2); os.dup2(w, 2)
    try:
        out, ms = run_pte(path, (x,), n=3)
    finally:
        os.dup2(saved, 2); os.close(w)
    err = os.read(r, 1 << 20).decode(errors="ignore"); os.close(r)
    ane = "ANE FAILED" if "ANECCompile() FAILED" in err else "ANE ok"
    globals_in = [i for i in [7, 15, 23, 31] if a <= i < b]
    print(f"layers {a}:{b} (global at {globals_in}) -> {ms:.0f} ms, {ane}, diff {(out[0]-ref).abs().max().item():.1e}", flush=True)
    os.remove(path)
