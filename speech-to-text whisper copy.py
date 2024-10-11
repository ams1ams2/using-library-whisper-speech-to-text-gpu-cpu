import whisper
import torch

# التأكد من أن GPU متاح
device = "cuda" if torch.cuda.is_available() else "cpu"

# تحميل النموذج
model = whisper.load_model("base").to(device)

# تحويل الصوت إلى نص
result = model.transcribe(r"out.wav")
print(result["text"])
