"""Export SAM 3 (image, text-prompted) to ExecuTorch as three programs.

  text.pte    (input_ids[1,32] int64, attention_mask[1,32] int64) -> text_features[1,32,256]
  vision.pte  (pixel_values[1,3,S,S] f32)                          -> fpn0, fpn1, fpn2
  head.pte    (fpn0, fpn1, fpn2, text_features, text_mask)         -> masks, boxes, logits, presence

Text runs once per prompt, vision + head per frame. FPN position encodings are constant for a
fixed input size, so they are baked into the head as buffers.
--random builds the default config with random weights (no gated download needed).
"""
import argparse, time
import torch
from torch import nn
from transformers import Sam3Config, Sam3Model


class Text(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m.get_text_features(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).pooler_output


class Vision(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.enc = m.vision_encoder

    def forward(self, pixel_values):
        out = self.enc(pixel_values)
        # The neck emits channels-last maps; Core ML only takes contiguous dim order.
        return tuple(f.contiguous() for f in out.fpn_hidden_states[:-1])


class Head(nn.Module):
    def __init__(self, m, pos):
        super().__init__()
        self.m = m
        # The decoder reads the feature-map size out of a tensor (data-dependent .item() under export).
        # The size is fixed for a fixed input, so pin it as Python ints.
        hw = tuple(int(x) for x in pos[-1].shape[-2:])
        dec = m.detr_decoder
        orig = type(dec)._get_rpb_matrix
        dec._get_rpb_matrix = lambda boxes, _shape: orig(dec, boxes, hw)
        for i, p in enumerate(pos):
            self.register_buffer(f"pos{i}", p.contiguous())
        self.n = len(pos)

    def forward(self, fpn0, fpn1, fpn2, text_features, text_mask):
        m = self.m
        fpn = [fpn0, fpn1, fpn2]
        pos = [getattr(self, f"pos{i}") for i in range(self.n)]
        mask = text_mask.bool()
        enc = m.detr_encoder(vision_features=[fpn[-1]], text_features=text_features, vision_pos_embeds=[pos[-1]], text_mask=mask)
        dec = m.detr_decoder(
            vision_features=enc.last_hidden_state, text_features=enc.text_features,
            vision_pos_encoding=enc.pos_embeds_flattened, text_mask=mask, spatial_shapes=enc.spatial_shapes,
        )
        offs = m.detr_decoder.box_head(dec.intermediate_hidden_states[-1:])
        ref = dec.reference_boxes[-1:].clamp(1e-5, 1 - 1e-5)
        boxes = (torch.log(ref / (1 - ref)) + offs).sigmoid()[-1]  # cxcywh, normalized
        logits = m.dot_product_scoring(
            decoder_hidden_states=dec.intermediate_hidden_states[-1:], text_features=enc.text_features, text_mask=mask
        ).squeeze(-1)[-1]
        masks = m.mask_decoder(
            decoder_queries=dec.intermediate_hidden_states[-1], backbone_features=fpn,
            encoder_hidden_states=enc.last_hidden_state, prompt_features=text_features, prompt_mask=mask,
        ).pred_masks
        return masks, boxes, logits, dec.presence_logits[-1]


def lower(ep, backend):
    from executorch.exir import to_edge_transform_and_lower
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        part = [XnnpackPartitioner()]
    elif backend == "coreml":
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        import os
        unit = getattr(ct.ComputeUnit, os.environ.get("COREML_UNITS", "ALL"))
        specs = CoreMLBackend.generate_compile_specs(compute_precision=ct.precision.FLOAT16, compute_unit=unit, minimum_deployment_target=ct.target.iOS17)
        part = [CoreMLPartitioner(compile_specs=specs)]
    else:
        part = []
    return to_edge_transform_and_lower(ep, partitioner=part).to_executorch()


def run_pte(path, args, n=3):
    from executorch.runtime import Runtime
    meth = Runtime.get().load_program(path).load_method("forward")
    out = meth.execute(list(args))
    t0 = time.time()
    for _ in range(n): meth.execute(list(args))
    return out, (time.time() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--parts", default="text,head,vision")
    ap.add_argument("--backend", default="xnnpack", choices=["xnnpack", "coreml", "portable", "none"])
    a = ap.parse_args()

    torch.manual_seed(0)
    if a.random:
        cfg = Sam3Config()
        cfg.vision_config.image_size = a.size
        cfg.vision_config.backbone_config.image_size = a.size
        model = Sam3Model(cfg).eval()
    else:
        cfg = Sam3Config.from_pretrained("facebook/sam3")
        cfg.vision_config.image_size = a.size; cfg.vision_config.backbone_config.image_size = a.size
        model = Sam3Model.from_pretrained("facebook/sam3", config=cfg).eval()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    ids = torch.randint(0, 49408, (1, 32)); am = torch.ones(1, 32, dtype=torch.long); am[:, 6:] = 0
    px = torch.randn(1, 3, a.size, a.size)
    with torch.no_grad():
        t0 = time.time(); vout = model.vision_encoder(px); tv = (time.time() - t0) * 1000
        fpn = tuple(f.contiguous() for f in vout.fpn_hidden_states[:-1]); pos = tuple(p.contiguous() for p in vout.fpn_position_encoding[:-1])
        tf = Text(model)(ids, am)
        head = Head(model, pos).eval()
        t0 = time.time(); ref = head(*fpn, tf, am); th = (time.time() - t0) * 1000
    print(f"eager on Mac CPU: vision {tv:.0f} ms, head {th:.0f} ms; fpn shapes {[tuple(f.shape) for f in fpn]}, text {tuple(tf.shape)}")
    print("head out shapes:", [tuple(r.shape) for r in ref])

    parts = {
        "text": (Text(model).eval(), (ids, am), tf if isinstance(tf, tuple) else (tf,)),
        "vision": (Vision(model).eval(), (px,), fpn),
        "head": (head, (*fpn, tf, am), ref),
    }
    for name in a.parts.split(","):
        mod, args, expected = parts[name]
        t0 = time.time()
        ep = torch.export.export(mod, args, strict=False)
        print(f"[{name}] torch.export ok ({time.time()-t0:.0f}s)")
        if a.backend == "none":
            continue
        prog = lower(ep, a.backend)
        out = f"sam3_{name}_{a.size}_{a.backend}{'_random' if a.random else ''}.pte"
        open(out, "wb").write(prog.buffer)
        print(f"[{name}] wrote {out} {len(prog.buffer)/1e6:.0f} MB")
        got, ms = run_pte(out, args, n=1 if name == "vision" else 3)
        diff = max((g - e).abs().max().item() for g, e in zip(got, expected))
        print(f"[{name}] pte vs eager max abs diff {diff:.2e}, {ms:.0f} ms/forward on Mac ({a.backend})")


if __name__ == "__main__":
    main()
