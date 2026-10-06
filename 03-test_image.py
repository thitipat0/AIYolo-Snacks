from ultralytics import YOLO
from pathlib import Path
import cv2
BASE_DIR = Path(__file__).resolve().parent

# โหลดโมเดลที่ผ่านการฝึก (Trained Model)
model = YOLO(str(BASE_DIR / "Model" / "best.pt"))
results = model.predict(str(BASE_DIR / "test.jpg"), conf=0.5, save=True)

# แสดงผลลัพธ์การตรวจจับของรูปภาพแรก


annotated = results[0].plot()
cv2.imshow("Result", annotated)
cv2.waitKey(0)        # press any key to close
cv2.destroyAllWindows()