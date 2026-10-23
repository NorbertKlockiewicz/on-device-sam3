# SAM 3 on iPhone

Meta's [SAM 3](https://huggingface.co/facebook/sam3) (848M parameters, text-prompted "segment every X") running fully on-device in a React Native app, with [react-native-executorch](https://github.com/software-mansion/react-native-executorch) and Core ML.

Take a photo, type a word, and every instance of it is segmented on the phone. No server, works in airplane mode.

## Numbers (iPhone 16 Pro, 672 px input)

| step | when | time |
|---|---|---|
| vision encoder (4 Core ML programs) | once per photo | ~1.5 to 2.5 s |
| text encoder | per prompt | ~20 ms |
| detection + mask head | per prompt | ~200 to 300 ms |

The vision encoder runs once per photo, so changing the prompt only costs the text encoder and the head. That is what makes type-as-you-go segmentation work.

Accuracy against PyTorch (same 672 px input, real weights): identical instances and scores on every test prompt, mask IoU 0.994 to 1.0. Against SAM 3 at its native 1008 px, 672 px finds the same instances on 6 of 7 test prompts (mask IoU 0.92 to 0.99); it can miss small, low-confidence objects.

## How it works

SAM 3 is split into three kinds of ExecuTorch programs:

- `sam3_text_672_coreml.pte`: CLIP text encoder, `[1,32]` token ids and mask to `[1,32,256]` features.
- `sam3_vision{0..3}_672_coreml.pte`: the 32-layer ViT in four 8-layer chunks plus the FPN neck. Apple's Neural Engine compiler rejects the whole 450M-parameter ViT as one program; 8-layer slices compile.
- `sam3_head_672_coreml.pte`: DETR encoder/decoder (200 queries), scoring and mask head. FPN position encodings are constant for a fixed input size and are baked in.

On the phone (`app/src/sam3.ts`): photo stretched to 672x672 (as SAM 3's processor does), vision once, then per prompt the tokenizer, text encoder and head run on a background worklet runtime. Scores are `sigmoid(logit) * sigmoid(presence)`, kept above 0.5. Mask logits are bilinearly upsampled 192 to 384 px before thresholding, and the coloured overlay is composited off the JS thread.

Export details, fixes and verification scripts are in [`ml/README.md`](ml/README.md).

## Run it

Requirements: Xcode 27, an iPhone with 8 GB RAM (iPhone 15 Pro or newer), a paid Apple developer team (the app needs the `increased-memory-limit` entitlement; the six models do not fit in the default per-app memory budget).

1. Point the app at the models. `.env.example` already points at the published exports on Hugging Face ([nklockiewicz/react-native-executorch-demo-models](https://huggingface.co/nklockiewicz/react-native-executorch-demo-models/tree/main/sam3)); to use your own exports instead, serve `ml/pte_real` and change the URL (see `ml/README.md`).
   ```bash
   cd app && cp .env.example .env
   ```
3. Build:
   ```bash
   cd app && npm install && npx expo prebuild --platform ios && npx expo run:ios --device
   ```

First launch downloads ~1.9 GB and compiles the Core ML models (~2 to 3 minutes). Later launches load in ~10 s.

## Licenses

Code in this repository: MIT (see `LICENSE`).

SAM 3 weights and anything derived from them, including the exported `.pte` files, are under Meta's [SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE). If you redistribute the exported models, include a copy of that license.
