from ultralytics import YOLO
from pathlib import Path
import torch
BASE_DIR = Path(__file__).resolve().parent

if __name__ == '__main__':
    model = YOLO('yolo26n.pt')
    results = model.train(
        data=str(BASE_DIR / "New-Snacks_dataset" / "data.yaml"),
        epochs=150,
        imgsz=640,
        optimizer="MuSGD",
        device=0 if torch.cuda.is_available() else "cpu",

        # --- พารามิเตอร์สำหรับ Data Augmentation หลัก ---
        degrees=7.0,        # สุ่มหมุนภาพ -15 ถึง +15 องศา (แก้ปัญหาภาพไม่เอียง)
        shear=5.0,           # สุ่มบิดภาพแบบเฉียงด้านขนาน (ช่วยจำลองมุมเอียง)
        perspective=0.001,   # เพิ่มความลึก (จำลองจากมุมที่ต่างออกไป)

        # --- การพลิกภาพ (Flip) ---
        fliplr=0.5,          # โอกาส 50% ที่จะพลิกภาพซ้าย-ขวา
        flipud=0.5,          # พลิกภาพบน-ล่าง (แนะนำให้ตั้งเป็น 0.0 เว้นแต่วัตถุของคุณในสภาพแวดล้อมจริงสามารถกลับด้านได้)

        # --- พารามิเตอร์ขั้นสูงที่แนะนำให้เปิดไว้ ---
        mosaic=0.1,          # (ค่าเริ่มต้น) นำภาพ 4 ภาพมาตัดแปรรวมกัน ช่วยให้โมเดลตรวจจับวัตถุขนาดเล็กได้ดีขึ้น
        mixup=0.1,           # ซ้อนภาพ 2 ภาพเข้าด้วยกันแบบโปร่งแสง ช่วยลด Overfitting ได้ดีมาก

        # ปิด Mosaic ในช่วง N epochs สุดท้ายเพื่อให้โมเดลปรับกับภาพจริง
        close_mosaic=10
    )