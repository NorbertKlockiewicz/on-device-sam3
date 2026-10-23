# Exporting SAM 3 to ExecuTorch

Python 3.12, [uv](https://docs.astral.sh/uv/). Weights come from the gated `facebook/sam3` repo on Hugging Face (request access, then `huggingface-cli login`).

```bash
uv sync
uv run export_vision_chunks.py --size 672                       # 4 vision chunks, Core ML
uv run export_sam3.py --backend coreml --size 672 --parts head,text
uv run verify_e2e.py                                            # full pipeline vs PyTorch on two photos
```

`--random` on either export script uses the default config with random weights, which is enough to test exportability and timing without the gated download.

## Scripts

- `export_sam3.py`: text encoder, head (and the whole vision encoder as one program, which only works on CPU/GPU).
- `export_vision_chunks.py`: vision encoder as four 8-layer Core ML programs (`COREML_UNITS=CPU_AND_NE` to restrict compute units).
- `verify_e2e.py`: runs text + 4 vision chunks + head through the ExecuTorch runtime and compares post-processed instances and masks with PyTorch.
- `accuracy_res.py`: SAM 3 at 672 px vs native 1008 px with real weights.
- `ane_probe.py`: which ViT layer slices the Apple Neural Engine compiler accepts, and their timing.

## Export notes

- The DETR decoder reads the feature-map size out of a tensor (`spatial_shapes[0, 0]`), which becomes a data-dependent `.item()` under `torch.export`. The size is fixed for a fixed input, so `_get_rpb_matrix` is pinned to Python ints.
- The vision neck emits channels-last tensors; Core ML only accepts contiguous dim order, so vision outputs and the baked position encodings are made contiguous.
- The whole ViT fails `ANECCompile()` at both 1008 and 672 px. Four 8-layer chunks compile. At 1008 px each global-attention layer costs ~300 ms on the Neural Engine, at 672 px ~60 ms, which is why the app uses 672.
- SAM 3 tiles its position embeddings, so other input sizes need no retraining.
- Raw mask-logit differences vs PyTorch look large (tens) but sit far from the threshold; thresholded masks match at IoU 0.994 to 1.0.
