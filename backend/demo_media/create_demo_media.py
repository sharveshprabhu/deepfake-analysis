import os
from pathlib import Path
from PIL import Image, ImageDraw

demo_dir = Path("demo_media")
demo_dir.mkdir(exist_ok=True)

# 1. Create a sample real image
real_img = Image.new("RGB", (320, 240), color=(70, 90, 120))
draw = ImageDraw.Draw(real_img)
draw.text((20, 20), "TruthLens Sample Real Image", fill=(255, 255, 255))
real_img.save(demo_dir / "sample_real.jpg")

# 2. Create sample fake image
fake_img = Image.new("RGB", (320, 240), color=(140, 50, 60))
draw = ImageDraw.Draw(fake_img)
draw.text((20, 20), "TruthLens Sample Fake Image", fill=(255, 255, 255))
fake_img.save(demo_dir / "sample_fake.jpg")

# 3. Create dummy video binary containers for offline testing
with open(demo_dir / "sample_real.mp4", "wb") as f:
    f.write(b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42" + b"REAL_VIDEO_STREAM_DATA" * 50)

with open(demo_dir / "sample_fake.mp4", "wb") as f:
    f.write(b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42" + b"DEEPFAKE_VIDEO_STREAM_DATA" * 50)

with open(demo_dir / "sample_no_audio.mp4", "wb") as f:
    f.write(b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42" + b"NO_AUDIO_VIDEO_DATA" * 50)

with open(demo_dir / "sample_inconclusive.mp4", "wb") as f:
    f.write(b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42" + b"DIFFICULT_INCONCLUSIVE_DATA" * 50)

print("Demo media created successfully in demo_media/")
