"""Phone-frame composite for the SAM 3 recording.

Redraws the provided frame (Desktop/PNG image (2).png, 346x714: 12px bezel, body 337 wide) at full resolution,
then builds the 1080x1350 video with ffmpeg: background -> screen recording (rounded corners) -> frame -> caption.
"""
import subprocess, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC = sys.argv[1]
OUT = sys.argv[2]
CAPTION = sys.argv[3] if len(sys.argv) > 3 else ""
W, H = 1080, 1350
SS = 3  # supersampling for anti-aliased edges

# Geometry from the reference frame (pixels in the 346x714 PNG).
REF_BODY_W, REF_BODY_H, REF_BEZEL = 337, 707, 12
REF_OUTER_R, REF_INNER_R = 55, 44

phone_h = 1170 if CAPTION else 1250
k = phone_h / REF_BODY_H
body_w, body_h = round(REF_BODY_W * k), round(phone_h)
bezel = REF_BEZEL * k
screen_w, screen_h = round(body_w - 2 * bezel), round(body_h - 2 * bezel)
x0 = (W - body_w) // 2
y0 = H - body_h - (60 if CAPTION else (H - body_h) // 2)
sx, sy = round(x0 + bezel), round(y0 + bezel)


def big(v):
    return round(v * SS)


# Background: soft vertical gradient.
bg = Image.new("RGB", (W, H))
d = ImageDraw.Draw(bg)
for y in range(H):
    t = y / H
    c = round(14 + 10 * (1 - abs(t - 0.45) * 2))
    d.line([(0, y), (W, y)], fill=(c, c, c + 4))
if CAPTION:
    font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 40)
    tw = d.textlength(CAPTION, font=font)
    d.text(((W - tw) / 2, 46), CAPTION, font=font, fill=(240, 240, 245))
bg.save("bg.png")

# Frame: bezel ring + side buttons, transparent screen. Drawn at SS x, downsampled.
fr = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
d = ImageDraw.Draw(fr)
btn = (40, 40, 42, 255)
for (side, top, length) in [("l", 150, 30), ("l", 205, 55), ("l", 270, 55), ("r", 225, 75)]:
    bx = x0 - 2.5 * k if side == "l" else x0 + body_w - 1.5 * k
    d.rounded_rectangle([big(bx), big(y0 + top * k), big(bx + 4 * k), big(y0 + (top + length) * k)], radius=big(2 * k), fill=btn)
d.rounded_rectangle([big(x0), big(y0), big(x0 + body_w), big(y0 + body_h)], radius=big(REF_OUTER_R * k), fill=(29, 29, 29, 255))
d.rounded_rectangle([big(x0 + 1.2 * k), big(y0 + 1.2 * k), big(x0 + body_w - 1.2 * k), big(y0 + body_h - 1.2 * k)],
                    radius=big((REF_OUTER_R - 1.2) * k), outline=(58, 58, 62, 255), width=big(0.8 * k))
d.rounded_rectangle([big(sx), big(sy), big(sx + screen_w), big(sy + screen_h)], radius=big(REF_INNER_R * k), fill=(0, 0, 0, 0))
fr = fr.resize((W, H), Image.LANCZOS)
fr.save("frame.png")

# Hide the screen-recording indicator: iOS expands the Dynamic Island with a red dot and glow. Paint the expanded
# pill over with the status-bar colour and draw a normal-size island (measured on the 1206x2622 recording).
probe = subprocess.run(
    ["ffmpeg", "-v", "error", "-ss", "1", "-i", SRC, "-frames:v", "1", "-vf", "crop=1206:200:0:0", "-f", "image2pipe", "-vcodec", "png", "-"],
    capture_output=True, check=True,
).stdout
import io
top = Image.open(io.BytesIO(probe)).convert("RGB")
status_bg = top.getpixel((300, 20))
patch = Image.new("RGBA", (1206 * SS, 200 * SS), (0, 0, 0, 0))
pd = ImageDraw.Draw(patch)
pd.rounded_rectangle([big(322), big(30), big(864), big(164)], radius=big(67), fill=status_bg + (255,))
pd.rounded_rectangle([big(418), big(40), big(788), big(154)], radius=big(57), fill=(0, 0, 0, 255))
patch.resize((1206, 200), Image.LANCZOS).save("island.png")

# Screen mask: rounded rect the size of the screen.
m = Image.new("L", (screen_w * SS, screen_h * SS), 0)
ImageDraw.Draw(m).rounded_rectangle([0, 0, screen_w * SS - 1, screen_h * SS - 1], radius=big(REF_INNER_R * k), fill=255)
m.resize((screen_w, screen_h), Image.LANCZOS).save("mask.png")

vf = (
    f"[1:v]fps=30,format=rgba[rec];[rec][4:v]overlay=0:0[recp];"
    f"[recp]scale={screen_w}:{screen_h}:flags=lanczos,format=rgba[scr];"
    f"[2:v]format=gray[m];[scr][m]alphamerge[scrm];"
    f"[0:v][scrm]overlay={sx}:{sy}[a];[a][3:v]overlay=0:0,format=yuv420p[v]"
)
dur = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", SRC],
    capture_output=True, text=True, check=True,
).stdout.strip()
cmd = [
    "ffmpeg", "-y", "-v", "error",
    "-loop", "1", "-i", "bg.png", "-i", SRC, "-loop", "1", "-i", "mask.png", "-loop", "1", "-i", "frame.png", "-loop", "1", "-i", "island.png",
    "-filter_complex", vf, "-map", "[v]", "-t", dur,
    "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart", "-an", OUT,
]
subprocess.run(cmd, check=True)
print("wrote", OUT, f"screen {screen_w}x{screen_h} at ({sx},{sy})")
