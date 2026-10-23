// SAM 3 on-device: text-prompted instance segmentation, split into 6 ExecuTorch programs.
//   vision0..3 run once per photo (~1.5 s on iPhone 16 Pro), text + head run per prompt (~0.2 s).
import { download, loadModel, tensor, wrapAsync, type Model, type Tensor } from 'react-native-executorch';
import { loadTokenizer, type Tokenizer } from 'react-native-executorch/nlp';
import type { ConcreteDim } from 'react-native-executorch/schema';

// Where the exported .pte files and tokenizer live (see ml/README.md). For local development, serve ml/pte_real
// with `python3 -m http.server 8766` and set EXPO_PUBLIC_SAM3_MODEL_URL=http://<your-mac>.local:8766 in app/.env.
const BASE = process.env.EXPO_PUBLIC_SAM3_MODEL_URL ?? '';
if (!BASE) throw new Error('Set EXPO_PUBLIC_SAM3_MODEL_URL in app/.env (see README.md)');
export const SIZE = 672;
const SEQ = 32;
const BOS = 49406;
const PAD = 49407; // CLIP pads with the end-of-text id
const SCORE_MIN = 0.5;
const MAX_INSTANCES = 24;
export const MASK_OUT = 384; // masks are bilinearly upsampled (as logits) to this size before thresholding

const FILES = {
  vision: [0, 1, 2, 3].map((i) => `${BASE}/sam3_vision${i}_${SIZE}_coreml.pte`),
  head: `${BASE}/sam3_head_${SIZE}_coreml.pte`,
  text: `${BASE}/sam3_text_${SIZE}_coreml.pte`,
  tokenizer: `${BASE}/sam3_tokenizer.json`,
};

export type Instance = {
  score: number;
  box: [number, number, number, number]; // x0, y0, x1, y1 in [0, 1]
};

// overlay: MASK_OUT x MASK_OUT RGBA, already composited (dimmed background, tinted matches, solid edges).
export type Segmentation = {
  prompt: string;
  instances: Instance[];
  overlay: Uint8Array;
  maskSize: number;
  textMs: number;
  headMs: number;
  overlayMs: number;
};

export const COLORS: [number, number, number][] = [
  [255, 64, 129], [0, 230, 118], [41, 182, 246], [255, 214, 0], [213, 0, 249],
  [255, 145, 0], [29, 233, 182], [255, 82, 82], [118, 255, 3], [124, 77, 255],
];
const DIM_ALPHA = 165;
const FILL_ALPHA = 95;

function dimValue(d: ConcreteDim): number {
  'worklet';
  if (d.kind === 'constant') return d.value;
  if (d.kind === 'enum') return d.choices[0]!;
  return d.range.min;
}

function allocOutputs(m: Model): Tensor[] {
  'worklet';
  return m.schema.forward!.outputs.map((p) => {
    if (p.kind !== 'Tensor') throw new Error('non-tensor output');
    return tensor(p.dtype, p.shape.map(dimValue));
  });
}

const sigmoid = (x: number) => {
  'worklet';
  return 1 / (1 + Math.exp(-x));
};

export async function downloadSam3(onProgress: (p: number) => void) {
  const all = [...FILES.vision, FILES.head, FILES.text, FILES.tokenizer];
  const paths = await download(all, { onProgress });
  return {
    vision: paths.slice(0, 4),
    head: paths[4]!,
    text: paths[5]!,
    tokenizer: paths[6]!,
  };
}

export type Sam3 = Awaited<ReturnType<typeof createSam3>>;

export async function createSam3(paths: Awaited<ReturnType<typeof downloadSam3>>, onStage?: (s: string) => void) {
  const loadRaw = wrapAsync(loadModel);
  const load = async (label: string, path: string) => {
    onStage?.(`compiling ${label}`);
    const t0 = Date.now();
    const m = await loadRaw(path);
    console.log(`[sam3] loaded ${label} in ${Date.now() - t0} ms`);
    return m;
  };
  const text = await load('text encoder', paths.text);
  const tokenizer: Tokenizer = await wrapAsync(loadTokenizer)(paths.tokenizer);
  const head = await load('head', paths.head);
  const vision: Model[] = [];
  for (let i = 0; i < 4; i++) vision.push(await load(`vision ${i + 1}/4`, paths.vision[i]!));

  // Pre-allocated native tensors, reused for every photo and prompt.
  const pixels = tensor('float32', [1, 3, SIZE, SIZE]);
  const visionOut = vision.map(allocOutputs); // chunks 0-2: hidden state, chunk 3: 3 FPN maps
  const ids = tensor('int64', [1, SEQ]);
  const mask = tensor('int64', [1, SEQ]);
  const [textFeatures] = allocOutputs(text);
  const headOut = allocOutputs(head); // masks [1,N,M,M], boxes [1,N,4], logits [1,N], presence [1,1]
  const maskSize = headOut[0]!.shape[3]!;
  const numQueries = headOut[0]!.shape[1]!;
  const oneMask = tensor('float32', [maskSize * maskSize]);

  const encodeWorklet = () => {
    'worklet';
    let h: Tensor = pixels;
    for (let i = 0; i < 4; i++) {
      vision[i]!.execute('forward', [h], visionOut[i]!);
      h = visionOut[i]![0]!;
    }
  };

  const segmentWorklet = (prompt: string) => {
    'worklet';
    const t0 = Date.now();
    let tok = Array.from(tokenizer.encode(prompt.toLowerCase()));
    if (tok[0] !== BOS) tok = [BOS, ...tok];
    if (tok[tok.length - 1] !== PAD) tok.push(PAD);
    tok = tok.slice(0, SEQ);
    const idsArr = new BigInt64Array(SEQ).fill(BigInt(PAD));
    const maskArr = new BigInt64Array(SEQ);
    tok.forEach((t, i) => {
      idsArr[i] = BigInt(t);
      maskArr[i] = 1n;
    });
    ids.setData(idsArr);
    mask.setData(maskArr);
    text.execute('forward', [ids, mask], [textFeatures!]);
    const t1 = Date.now();
    const [fpn0, fpn1, fpn2] = visionOut[3]!;
    head.execute('forward', [fpn0!, fpn1!, fpn2!, textFeatures!, mask], headOut);
    const t2 = Date.now();

    const boxes = headOut[1]!.getData(new Float32Array(numQueries * 4));
    const logits = headOut[2]!.getData(new Float32Array(numQueries));
    const presence = sigmoid(headOut[3]!.getData(new Float32Array(1))[0]!);
    const keep: { i: number; score: number }[] = [];
    for (let i = 0; i < numQueries; i++) {
      const score = sigmoid(logits[i]!) * presence;
      if (score >= SCORE_MIN) keep.push({ i, score });
    }
    keep.sort((a, b) => b.score - a.score);

    const buf = new Float32Array(maskSize * maskSize);
    const n = maskSize;
    const O = MASK_OUT;
    const scale = n / O;
    const owner = new Int16Array(O * O).fill(-1);
    const kept = keep.slice(0, MAX_INSTANCES);
    const instances: Instance[] = kept.map(({ i, score }, k) => {
      headOut[0]!.copyTo(oneMask, { offset: i * n * n, length: n * n });
      oneMask.getData(buf);
      // Bilinear upsample of the logits, then threshold: smooth edges instead of 192px stair steps.
      // Higher-scoring instances claim pixels first.
      for (let y = 0; y < O; y++) {
        const fy = Math.min(n - 1, Math.max(0, (y + 0.5) * scale - 0.5));
        const y0 = Math.floor(fy);
        const y1 = Math.min(n - 1, y0 + 1);
        const wy = fy - y0;
        for (let x = 0; x < O; x++) {
          const fx = Math.min(n - 1, Math.max(0, (x + 0.5) * scale - 0.5));
          const x0 = Math.floor(fx);
          const x1 = Math.min(n - 1, x0 + 1);
          const wx = fx - x0;
          const v =
            (buf[y0 * n + x0]! * (1 - wx) + buf[y0 * n + x1]! * wx) * (1 - wy) +
            (buf[y1 * n + x0]! * (1 - wx) + buf[y1 * n + x1]! * wx) * wy;
          if (v > 0 && owner[y * O + x] === -1) owner[y * O + x] = k;
        }
      }
      const [cx, cy, w, h] = [boxes[i * 4]!, boxes[i * 4 + 1]!, boxes[i * 4 + 2]!, boxes[i * 4 + 3]!];
      return { score, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] };
    });

    // Composite overlay here, off the JS thread.
    const overlay = new Uint8Array(O * O * 4);
    const any = instances.length > 0;
    for (let y = 0; y < O; y++) {
      for (let x = 0; x < O; x++) {
        const p = y * O + x;
        const k = owner[p]!;
        if (k < 0) {
          if (any) overlay[p * 4 + 3] = DIM_ALPHA;
          continue;
        }
        const c = COLORS[k % COLORS.length]!;
        const edge =
          x === 0 || y === 0 || x === O - 1 || y === O - 1 ||
          owner[p - 1] !== k || owner[p + 1] !== k || owner[p - O] !== k || owner[p + O] !== k;
        overlay[p * 4] = c[0];
        overlay[p * 4 + 1] = c[1];
        overlay[p * 4 + 2] = c[2];
        overlay[p * 4 + 3] = edge ? 255 : FILL_ALPHA;
      }
    }
    const t3 = Date.now();
    return { prompt, instances, overlay, maskSize: O, textMs: t1 - t0, headMs: t2 - t1, overlayMs: t3 - t2 };
  };

  const encodeAsync = wrapAsync(encodeWorklet);
  const segmentAsync = wrapAsync(segmentWorklet);
  // First executions are slow (Core ML warm-up, ~1 s each); pay them at load on a blank photo, not on camera.
  onStage?.('warming up');
  const tw = Date.now();
  await encodeAsync();
  await segmentAsync('warm up');
  console.log(`[sam3] warm-up ${Date.now() - tw} ms`);
  const chw = new Float32Array(3 * SIZE * SIZE);

  return {
    maskSize,
    /** rgba: SIZE x SIZE x 4 bytes. Fills the pixel tensor on the JS thread, then runs the ViT in the background. */
    async encode(rgba: Uint8Array): Promise<number> {
      const plane = SIZE * SIZE;
      for (let p = 0; p < plane; p++) {
        chw[p] = rgba[p * 4]! / 127.5 - 1;
        chw[plane + p] = rgba[p * 4 + 1]! / 127.5 - 1;
        chw[2 * plane + p] = rgba[p * 4 + 2]! / 127.5 - 1;
      }
      pixels.setData(chw);
      const t0 = Date.now();
      await encodeAsync();
      return Date.now() - t0;
    },
    segment: (prompt: string): Promise<Segmentation> => segmentAsync(prompt),
  };
}
