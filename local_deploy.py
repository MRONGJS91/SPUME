# ========================= 100% 能跑的最终版 =========================
from fastapi import FastAPI, UploadFile
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import io
import torch
import os

# 强制离线，不联网、不报错
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 你的模型路径（你已经测试成功的路径）
MODEL_PATH = r"D:\SPUME\vit-gpt2-model"

# ✅ 必须写这一行！修复你报错的核心
app = FastAPI()

# 加载本地模型
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 接口
@app.post("/caption")
async def run_caption(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(image, return_tensors="pt").pixel_values.to(device)
    output = model.generate(inputs, max_length=32)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"caption": caption}