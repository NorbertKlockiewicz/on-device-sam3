import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  Keyboard,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  type LayoutChangeEvent,
} from 'react-native';
import {
  AlphaType,
  Canvas,
  ColorType,
  FilterMode,
  Image as SkiaImage,
  MipmapMode,
  Skia,
  type SkImage,
} from '@shopify/react-native-skia';
import * as ImagePicker from 'expo-image-picker';
import {
  Camera,
  CommonResolutions,
  useCameraDevice,
  useCameraPermission,
  usePhotoOutput,
} from 'react-native-vision-camera';
import { SIZE, createSam3, downloadSam3, type Sam3, type Segmentation } from './src/sam3';

const SUGGESTIONS = ['person', 'chair', 'book', 'plant', 'vase', 'cup', 'pen', 'phone', 'hand', 'cat'];
// Optional comma-separated sample image URLs for quick testing without a camera.
const SAMPLES = (process.env.EXPO_PUBLIC_SAM3_SAMPLES ?? '').split(',').filter(Boolean);
const DEBOUNCE_MS = 120;

// Load SAM 3 once per process: Fast Refresh re-runs effects, and a second copy of the models (~2.5 GB) gets the app killed.
const g = globalThis as { __sam3?: Promise<Sam3> };
function loadOnce(onStatus: (s: string) => void): Promise<Sam3> {
  g.__sam3 ??= (async () => {
    const paths = await downloadSam3((p) => onStatus(`Downloading SAM 3 · ${Math.round(p * 100)}%`));
    const t0 = Date.now();
    const s = await createSam3(paths, (st) => onStatus(`Preparing · ${st}`));
    console.log(`[sam3] loaded in ${Date.now() - t0} ms, mask ${s.maskSize}px`);
    return s;
  })();
  return g.__sam3;
}
const GUTTER = 16;
const FRAME_GUTTER = 8;
const ROOT_BOTTOM_PAD = 24;

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const photoOutput = usePhotoOutput({
    targetResolution: CommonResolutions.HD_4_3,
    containerFormat: 'jpeg',
    qualityPrioritization: 'speed',
  });

  const [status, setStatus] = useState('downloading SAM 3');
  const [sam, setSam] = useState<Sam3 | null>(null);
  const [photo, setPhoto] = useState<SkImage | null>(null);
  const [encoding, setEncoding] = useState(false);
  const [visionMs, setVisionMs] = useState<number | null>(null);
  const [prompt, setPrompt] = useState('');
  const [seg, setSeg] = useState<Segmentation | null>(null);
  const [stage, setStage] = useState({ w: 0, h: 0 });
  const busy = useRef(false);
  const lift = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const show = Keyboard.addListener('keyboardWillShow', (e) =>
      Animated.timing(lift, {
        toValue: -(e.endCoordinates.height - ROOT_BOTTOM_PAD),
        duration: e.duration || 250,
        useNativeDriver: true,
      }).start()
    );
    const hide = Keyboard.addListener('keyboardWillHide', (e) =>
      Animated.timing(lift, { toValue: 0, duration: e.duration || 250, useNativeDriver: true }).start()
    );
    return () => {
      show.remove();
      hide.remove();
    };
  }, [lift]);
  const queued = useRef<string | null>(null);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    loadOnce(setStatus)
      .then(setSam)
      .catch((e) => {
        g.__sam3 = undefined;
        console.log('[sam3] load error', String(e));
        setStatus(`Error · ${String(e)}`);
      });
  }, []);

  const capture = async () => {
    if (!sam || encoding) return;
    const file = await photoOutput.capturePhotoToFile({ flashMode: 'off', enableShutterSound: false }, {});
    await loadPhoto(`file://${file.filePath}`);
  };

  const pick = async () => {
    if (!sam || encoding) return;
    // Library photos are usually HEIC, which Skia cannot decode; 'compatible' makes iOS hand over a JPEG.
    const r = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 0.95,
      preferredAssetRepresentationMode: ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Compatible,
    });
    if (r.canceled || !r.assets[0]) return;
    try {
      await loadPhoto(r.assets[0].uri);
    } catch (e) {
      console.log('[sam3] pick error', String(e), r.assets[0].uri);
      setStatus(`Could not open that photo · ${String(e)}`);
    }
  };

  const sampleIdx = useRef(0);
  const sample = async () => {
    if (!sam || encoding) return;
    const uri = SAMPLES[sampleIdx.current++ % SAMPLES.length]!;
    await loadPhoto(uri);
  };

  const loadPhoto = async (uri: string) => {
    if (!sam) return;
    const data = await Skia.Data.fromURI(uri);
    const img = Skia.Image.MakeImageFromEncoded(data);
    if (!img) throw new Error('could not decode photo');
    // SAM 3's processor stretches any photo to its square input, so do the same and stretch the masks back.
    const surface = Skia.Surface.MakeOffscreen(SIZE, SIZE)!;
    surface
      .getCanvas()
      .drawImageRect(img, Skia.XYWHRect(0, 0, img.width(), img.height()), Skia.XYWHRect(0, 0, SIZE, SIZE), Skia.Paint());
    surface.flush();
    const square = surface.makeImageSnapshot().makeNonTextureImage()!;
    setPhoto(img.makeNonTextureImage() ?? img);
    setSeg(null);
    setEncoding(true);
    const ms = await sam.encode(square.readPixels() as Uint8Array);
    console.log(`[sam3] vision ${ms} ms (${img.width()}x${img.height()})`);
    setVisionMs(ms);
    setEncoding(false);
  };

  const run = async (p: string) => {
    if (!sam || !photo || !p.trim()) {
      setSeg(null);
      return;
    }
    if (busy.current) {
      queued.current = p;
      return;
    }
    busy.current = true;
    try {
      const r = await sam.segment(p.trim());
      console.log(`[sam3] '${p}' -> ${r.instances.length} inst, text ${r.textMs} ms, head ${r.headMs} ms, overlay ${r.overlayMs} ms`);
      setSeg(r);
    } catch (e) {
      console.log('[sam3] segment error', String(e));
    } finally {
      busy.current = false;
      const next = queued.current;
      queued.current = null;
      if (next !== null && next !== p) run(next);
    }
  };

  useEffect(() => {
    if (encoding) return;
    const t = setTimeout(() => run(prompt), DEBOUNCE_MS);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prompt, photo, encoding]);

  const overlay = useMemo(() => {
    if (!seg || !prompt.trim()) return null;
    const n = seg.maskSize;
    return Skia.Image.MakeImage(
      { width: n, height: n, alphaType: AlphaType.Unpremul, colorType: ColorType.RGBA_8888 },
      Skia.Data.fromBytes(seg.overlay),
      n * 4
    );
  }, [seg, prompt]);

  // Full-width frame at the photo's aspect (3:4 for the camera). The keyboard slides over it; the photo never resizes.
  const aspect = photo ? photo.height() / photo.width() : 4 / 3;
  let W = Math.max(0, stage.w - 2 * FRAME_GUTTER);
  let H = W * aspect;
  if (H > stage.h) {
    H = Math.max(0, stage.h);
    W = H / aspect;
  }
  const onStage = (e: LayoutChangeEvent) => setStage({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height });
  const count = seg?.instances.length ?? 0;
  const ready = !!sam;

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <Text style={styles.title}>Segment Anything</Text>
        <Text style={styles.badge}>SAM 3 · 848M · on-device</Text>
      </View>

      <Pressable style={styles.stage} onLayout={onStage} onPress={Keyboard.dismiss} accessible={false}>
        <View style={[styles.frame, { width: W, height: H }]}>
          {!photo && device && hasPermission && (
            <Camera style={StyleSheet.absoluteFill} device={device} isActive outputs={[photoOutput]} resizeMode="cover" />
          )}
          {W > 0 && (
            <Canvas style={StyleSheet.absoluteFill} pointerEvents="none">
              {photo && <SkiaImage image={photo} x={0} y={0} width={W} height={H} fit="fill" />}
              {overlay && (
                <SkiaImage
                  image={overlay}
                  x={0}
                  y={0}
                  width={W}
                  height={H}
                  fit="fill"
                  sampling={{ filter: FilterMode.Linear, mipmap: MipmapMode.None }}
                />
              )}
            </Canvas>
          )}
          {!photo && ready && !encoding && SAMPLES.length > 0 && (
            <Pressable onPress={sample} accessibilityRole="button" accessibilityLabel="Try a sample" style={styles.samplePill}>
              <Text style={styles.samplePillText}>Try a sample</Text>
            </Pressable>
          )}
          {(encoding || !ready) && (
            <View style={styles.veil}>
              <Text style={styles.veilText}>{ready ? 'Reading the photo…' : status}</Text>
            </View>
          )}
          {photo && seg && !encoding && prompt.trim() !== '' && (
            <View style={styles.countPill}>
              <Text style={styles.countText}>
                {count === 0 ? `no ${seg.prompt}` : `${count} × ${seg.prompt}`}
              </Text>
            </View>
          )}
        </View>
      </Pressable>

      <Animated.View style={[styles.bottom, { transform: [{ translateY: lift }] }]}>
        {photo && !encoding && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.chips} keyboardShouldPersistTaps="always">
            {SUGGESTIONS.map((s) => (
              <Pressable key={s} onPress={() => setPrompt(s)} accessibilityRole="button" accessibilityLabel={`Suggest ${s}`} style={[styles.chip, prompt === s && styles.chipOn]}>
                <Text style={[styles.chipText, prompt === s && styles.chipTextOn]}>{s}</Text>
              </Pressable>
            ))}
          </ScrollView>
        )}
        <View style={styles.bar}>
          <Pressable
            onPress={photo ? () => { setPhoto(null); setSeg(null); } : capture}
            disabled={!ready || encoding}
            accessibilityRole="button"
            accessibilityLabel={photo ? 'Retake' : 'Take photo'}
            style={[styles.shutter, (!ready || encoding) && styles.off]}
          >
            <View style={photo ? styles.shutterRetake : styles.shutterDot} />
          </Pressable>
          <Pressable onPress={pick} disabled={!ready || encoding} accessibilityRole="button" accessibilityLabel="Pick from library" style={[styles.gallery, (!ready || encoding) && styles.off]}>
            <View style={styles.galleryIcon} />
          </Pressable>
          <TextInput
            style={styles.input}
            value={prompt}
            onChangeText={setPrompt}
            placeholder={photo ? 'Find anything…' : 'Take a photo first'}
            placeholderTextColor="#6b6b73"
            autoCapitalize="none"
            autoCorrect={false}
            returnKeyType="search"
            onSubmitEditing={Keyboard.dismiss}
            blurOnSubmit
            editable={ready && !!photo && !encoding}
            clearButtonMode="while-editing"
            accessibilityLabel="Prompt"
          />
        </View>
        <Text style={styles.stats}>
          {visionMs != null ? `photo ${(visionMs / 1000).toFixed(1)} s` : 'photo –'}
          {'  ·  '}
          {seg ? `prompt ${seg.textMs + seg.headMs} ms` : 'prompt –'}
          {'  ·  airplane-mode safe'}
        </Text>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0a0c', paddingTop: 54, paddingBottom: ROOT_BOTTOM_PAD },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: GUTTER, marginBottom: 8 },
  title: { color: '#fff', fontSize: 22, fontWeight: '700', letterSpacing: -0.3 },
  badge: { color: '#9a9aa3', fontSize: 12, fontVariant: ['tabular-nums'] },
  stage: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  frame: { borderRadius: 22, overflow: 'hidden', backgroundColor: '#141418' },
  veil: { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.45)', alignItems: 'center', justifyContent: 'center' },
  veilText: { color: '#fff', fontSize: 15, fontWeight: '600' },
  countPill: { position: 'absolute', top: 14, left: 14, backgroundColor: 'rgba(255,255,255,0.92)', paddingHorizontal: 14, paddingVertical: 7, borderRadius: 999 },
  countText: { color: '#0a0a0c', fontSize: 16, fontWeight: '700', fontVariant: ['tabular-nums'] },
  bottom: { paddingTop: 10, paddingBottom: 6, backgroundColor: 'rgba(10,10,12,0.85)' },
  samplePill: { position: 'absolute', bottom: 14, alignSelf: 'center', backgroundColor: 'rgba(0,0,0,0.55)', paddingHorizontal: 16, paddingVertical: 8, borderRadius: 999, borderWidth: 1, borderColor: 'rgba(255,255,255,0.25)' },
  samplePillText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  chips: { paddingHorizontal: GUTTER, gap: 8, paddingBottom: 10 },
  chip: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: 999, backgroundColor: '#1c1c21' },
  chipOn: { backgroundColor: '#fff' },
  chipText: { color: '#d5d5db', fontSize: 14, fontWeight: '600' },
  chipTextOn: { color: '#0a0a0c' },
  bar: { flexDirection: 'row', alignItems: 'center', gap: 10, marginHorizontal: GUTTER },
  shutter: { width: 52, height: 52, borderRadius: 26, borderWidth: 3, borderColor: '#fff', alignItems: 'center', justifyContent: 'center' },
  shutterDot: { width: 38, height: 38, borderRadius: 19, backgroundColor: '#fff' },
  shutterRetake: { width: 18, height: 18, borderRadius: 4, backgroundColor: '#fff' },
  off: { opacity: 0.35 },
  gallery: { width: 52, height: 52, borderRadius: 26, backgroundColor: '#1c1c21', alignItems: 'center', justifyContent: 'center' },
  galleryIcon: { width: 22, height: 18, borderRadius: 4, borderWidth: 2.5, borderColor: '#fff' },
  input: { flex: 1, height: 52, color: '#fff', fontSize: 18, paddingHorizontal: 18, backgroundColor: '#1c1c21', borderRadius: 26 },
  stats: { color: '#6b6b73', fontSize: 12, textAlign: 'center', marginTop: 10, fontVariant: ['tabular-nums'] },
});
