from ultralytics import YOLO
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

model = YOLO(str(BASE_DIR / "Model" / "best.pt"))
video_to_test = str(BASE_DIR / "VID_20261005_230813.mp4")

results = model.predict(str(BASE_DIR / video_to_test), conf=0.5, save=True, show=True)

print("ทดสอบเสร็จสิ้น! สามารถเข้าดูวิดีโอผลลัพธ์ได้ที่โฟลเดอร์ runs/detect/predict")