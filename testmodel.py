from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# 本地模型路径
MODEL_PATH = r"D:\SPUME\vit-gpt2-model"

# 加载模型、处理器、分词器
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
image_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("✅ 模型加载成功！")