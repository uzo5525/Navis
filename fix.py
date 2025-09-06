from PIL import Image, ImageOps
import os, glob

SRC = r"E:\raw_dataset"         # 원본 이미지 폴더
DST = r"E:\raw_dataset"     # 보정본 저장 폴더
os.makedirs(DST, exist_ok=True)

exts = ("*.jpg","*.jpeg","*.png","*.webp")
files = []
for e in exts: files += glob.glob(os.path.join(SRC, e))

for f in files:
    img = Image.open(f)
    img = ImageOps.exif_transpose(img)   # EXIF 기준으로 실제 픽셀 회전
    img.save(os.path.join(DST, os.path.basename(f)))
print("done:", len(files))
