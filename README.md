# 🍿 AIYolo-Snacks: ระบบตรวจจับและจำแนกวัตถุซองขนมด้วย YOLO26

![YOLO26 Detection](https://img.shields.io/badge/YOLO-v8%20%2F%20v11%20%2F%20YOLO26-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Supported-orange)
![Label Studio](https://img.shields.io/badge/Annotation-Label%20Studio-brightgreen)

โปรเจกต์พัฒนาระบบ Computer Vision เพื่อการตรวจจับ (Object Detection) และจำแนกชนิดซองขนม 3 ยี่ห้อ (**Oreo**, **Lausanne**, **Voiz**) แบบอัตโนมัติ โดยรองรับการทำงานทั้งแบบออฟไลน์ (รูปภาพ/วิดีโอ) และแบบ Real-time ผ่านกล้องเว็บแคม เหมาะสำหรับประยุกต์ใช้ในระบบตรวจนับสินค้าหน้าร้าน (Retail Automation) หรือระบบการจัดการคลังสินค้าอัตโนมัติ

---

## 📌 สารบัญ (Table of Contents)
1. [คุณสมบัติของระบบ (Key Features)](#-คุณสมบัติของระบบ-key-features)
2. [คลาสเป้าหมาย (Target Classes)](#-คลาสเป้าหมาย-target-classes)
3. [โครงสร้างโปรเจกต์ (Project Structure)](#-โครงสร้างโปรเจกต์-project-structure)
4. [ข้อกำหนดและการติดตั้ง (Requirements & Installation)](#-ข้อกำหนดและการติดตั้ง-requirements--installation)
5. [ขั้นตอนการทำงานอย่างละเอียด (End-to-End Pipeline)](#-ขั้นตอนการทำงานอย่างละเอียด-end-to-end-pipeline)
   - [1. การจัดเตรียมรูปภาพและการดึงเฟรม (Data Preparation)](#1-การจัดเตรียมรูปภาพและการดึงเฟรม-data-preparation)
   - [2. การทำ Annotation ด้วย Label Studio](#2-การทำ-annotation-ด้วย-label-studio)
   - [3. การแปลงชุดข้อมูลเป็น YOLO Format](#3-การแปลงชุดข้อมูลเป็น-yolo-format)
   - [4. การเทรนโมเดล (Model Training)](#4-การเทรนโมเดล-model-training)
   - [5. การประเมินผลและการทดสอบ (Evaluation & Inference)](#5-การประเมินผลและการทดสอบ-evaluation--inference)
6. [การตั้งค่าพารามิเตอร์ไฮเปอร์พารามิเตอร์ (Hyperparameters)](#-การตั้งค่าพารามิเตอร์ไฮเปอร์พารามิเตอร์-hyperparameters)
7. [การแก้ไขปัญหาที่พบบ่อย (Troubleshooting & FAQs)](#-การแก้ไขปัญหาที่พบบ่อย-troubleshooting--faqs)

---

## 🚀 คุณสมบัติของระบบ (Key Features)

- **High-Accuracy Detection:** ใช้สถาปัตยกรรม YOLO26 ล่าสุดในการจับขอบเขต (Bounding Box) และระบุคลาสได้อย่างแม่นยำแม้ในสภาวะแสงที่ต่างกัน
- **Real-time Performance:** รองรับการประมวลผลผ่านกล้องเว็บแคมด้วยเฟรมเรตสูงผ่านการเร่งประมวลผลด้วย GPU (CUDA Acceleration)
- **Robust Augmentation:** มีระบบการทำ Data Augmentation ทั้ง Mosaic, Mixup, Random Rotation และ Perspective Shift ช่วยให้โมเดลไม่ Overfit
- **Automated Dataset Conversion:** มีสคริปต์แปลงไฟล์ Annotation (JSON) จาก Label Studio เข้าสู่โครงสร้าง YOLO Dataset โดยอัตโนมัติ พร้อมสุ่มแบ่ง Split Data (Train/Val) แบบควบคุม Seed ได้

---

## 🏷 คลาสเป้าหมาย (Target Classes)

โมเดลถูกฝึกสอนให้ตรวจจับวัตถุขนมทั้งหมด 3 ชนิด ดังนี้:

| Class ID | Class Name | Description | Color Code (Label Studio) |
| :---: | :---: | :--- | :---: |
| `0` | **Oreo** | ซองขนมโอรีโอ (คุกกี้สอดไส้ครีม) | `#FF0000` (Red) |
| `1` | **Lausanne** | ซองขนมลอซานน์ (เวเฟอร์) | `#00AA00` (Green) |
| `2` | **Voiz** | ซองขนมวอยซ์ (แครกเกอร์/เวเฟอร์) | `#0000FF` (Blue) |

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
AIYolo-Snacks/
├── Model/
│   └── best.pt                           # ไฟล์ Weight ของโมเดลที่เทรนสำเร็จแล้ว (Best Weights)
├── New-Snacks_dataset/                   # ชุดข้อมูลรูปภาพและ Labels ในรูปแบบ YOLO Format
│   ├── images/
│   │   ├── train/                        # รูปภาพสำหรับการเทรน
│   │   └── val/                          # รูปภาพสำหรับการตรวจสอบ (Validation)
│   ├── labels/
│   │   ├── train/                        # ไฟล์ .txt Bounding box สำหรับ Train
│   │   └── val/                          # ไฟล์ .txt Bounding box สำหรับ Val
│   └── data.yaml                         # ไฟล์คอนฟิกูเรชันอธิบาย Path และ Classes ของ Dataset
├── frame/                                # โฟลเดอร์เก็บเฟรม/รูปภาพตั้งต้นสำหรับนำเข้า Label Studio
├── 01-export_dataset.py                  # สคริปต์สกัด JSON จาก Label Studio เป็น YOLO Format
├── 02-train2.py                          # สคริปต์สำหรับการเทรนโมเดล YOLO
├── 03-test_image.py                      # สคริปต์ทดสอบการตรวจจับกับไฟล์รูปภาพเดี่ยว/โฟลเดอร์
├── 04-test_video.py                      # สคริปต์ทดสอบการตรวจจับกับไฟล์วิดีโอ (.mp4, .avi)
├── 05-test-camera2.py                    # สคริปต์ตรวจจับแบบ Real-time ผ่านกล้อง Webcam
├── project-9-at-2026-10-05-23-47-6...json # ไฟล์ Export ข้อมูล Label ทั้งหมดจาก Label Studio
├── requirements.txt                      # รายการ Dependencies และ ไลบรารีที่ต้องใช้
├── test.jpg                              # รูปภาพตัวอย่างสำหรับทดสอบการทำงานรวดเร็ว
└── README.md                             # คู่มือและเอกสารกำกับโปรเจกต์
```

---

## 🛠 ข้อกำหนดและการติดตั้ง (Requirements & Installation)

### 1. สภาพแวดล้อมระบบ (Environment Setup)
แนะนำให้ใช้ **Python 3.9 - 3.11** และทำงานผ่าน Virtual Environment เพื่อป้องกันความขัดแย้งของไลบรารี

```bash
# 1. สร้าง Virtual Environment
python -m venv env

# 2. เปิดใช้งาน Virtual Environment
# สำหรับ Windows (Command Prompt)
env\Scripts\activate

# สำหรับ Windows (PowerShell)
.\env\Scripts\Activate.ps1

# สำหรับ Linux / macOS
source env/bin/activate
```

### 2. ติดตั้ง PyTorch พร้อมการรองรับ GPU (CUDA)
เพื่อให้ระบบประมวลผลได้อย่างรวดเร็ว ควรติดตั้ง PyTorch เวอร์ชันรองรับ CUDA (ตัวอย่างสำหรับ CUDA 12.1):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

*สามารถตรวจสอบสถานะการเชื่อมต่อ GPU ได้ด้วยการรันคำสั่ง:*
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. ติดตั้ง Dependencies อื่นๆ
```bash
pip install -r requirements.txt
```

---

## 🔄 ขั้นตอนการทำงานอย่างละเอียด (End-to-End Pipeline)

```text
  ┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
  │  1. Prep Frames │ ───> │  2. Label Studio │ ───> │ 3. Export JSON File │
  └─────────────────┘      └──────────────────┘      └─────────────────────┘
                                                                │
  ┌─────────────────┐      ┌──────────────────┐                 ▼
  │ 5. Test/Infer   │ <─── │ 4. Train Model   │ <─── ┌─────────────────────┐
  │ (Image/Cam/Vid) │      │  (02-train2.py)  │      │01-export_dataset.py │
  └─────────────────┘      └──────────────────┘      └─────────────────────┘
```

### 1. การจัดเตรียมรูปภาพและการดึงเฟรม (Data Preparation)
- นำรูปภาพซองขนม **Oreo**, **Lausanne**, และ **Voiz** ที่บันทึกได้ มาจัดเก็บไว้ในโฟลเดอร์ `frame/`
- แนะนำให้ถ่ายภาพในหลายๆ มุมมอง, สภาพแสงที่แตกต่างกัน, และมีการวางซ้อนทับกันบางส่วนเพื่อสร้าง Dataset ที่ครอบคลุม

### 2. การทำ Annotation ด้วย Label Studio
1. เปิดใช้งาน Label Studio:
   ```bash
   label-studio start
   ```
2. สร้าง Project ใหม่ และกำหนด **Labeling Interface** ในส่วน Custom XML ดังนี้:
   ```xml
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label background="#FF0000" value="Oreo"/>
       <Label background="#00AA00" value="Lausanne"/>
       <Label background="#0000FF" value="Voiz"/>
     </RectangleLabels>
   </View>
   ```
3. ทำการวาด Bounding Box ครอบซองขนมทุกชิ้นในภาพให้ชิดขอบวัตถุมากที่สุด
4. เมื่อทำเสร็จแล้ว ให้กด **Export** เลือกรูปแบบ **JSON** แล้วนำไฟล์ `.json` ที่ได้มาวางไว้ที่ Root Directory ของโปรเจกต์ (เช่น `project-9-at-2026-10-05-23-47-6...json`)

### 3. การแปลงชุดข้อมูลเป็น YOLO Format
รันสคริปต์ `01-export_dataset.py` เพื่ออ่านไฟล์ JSON และแปลงพิกัดวัตถุพร้อมแบ่งชุดข้อมูลเป็น Train (85%) และ Validation (15%):

```bash
python 01-export_dataset.py
```
*ผลลัพธ์จะถูกสร้างไว้ในโฟลเดอร์ `New-Snacks_dataset/` พร้อมไฟล์ `data.yaml`*

### 4. การเทรนโมเดล (Model Training)
เริ่มการฝึกสอนโมเดลโดยใช้สคริปต์ `02-train2.py`:

```bash
python 02-train2.py
```
การทำงานของสคริปต์จะทำการบันทึก Log และ Weights ไว้ในโฟลเดอร์ `runs/detect/` เมื่อการเทรนสิ้นสุดลง ให้นำไฟล์ `best.pt` คัดลอกมาไว้ที่โฟลเดอร์ `Model/best.pt` เพื่อนำไปใช้งานต่อ

### 5. การประเมินผลและการทดสอบ (Evaluation & Inference)

- **การทดสอบกับรูปภาพเดี่ยว/โฟลเดอร์:**
  ```bash
  python 03-test_image.py
  ```
  *(ปรับแก้ Path รูปภาพที่ต้องการทดสอบในไฟล์สคริปต์)*

- **การทดสอบกับไฟล์วิดีโอ:**
  ```bash
  python 04-test_video.py
  ```

- **การทดสอบตรวจจับ Real-time ผ่านกล้องเว็บแคม:**
  ```bash
  python 05-test-camera2.py
  ```
  *(กดปุ่ม `q` บนคีย์บอร์ดเมื่อต้องการปิดหน้าต่างการทำงานของกล้อง)*

---

## ⚙️ การตั้งค่าพารามิเตอร์ไฮเปอร์พารามิเตอร์ (Hyperparameters)

ค่าพารามิเตอร์หลักที่ถูกกำหนดไว้ในสคริปต์ `02-train2.py` สำหรับปรับแต่งประสิทธิภาพของโมเดล:

| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| **Base Model** | `yolo26n.pt` / `yolov8n.pt` | โมเดลเริ่มต้นที่ใช้ทำ Transfer Learning |
| **Epochs** | `200` | จำนวนรอบทั้งหมดในการฝึกสอน |
| **Patience** | `30` | ปรับ Early Stopping เมื่อ Metric ไม่ดีขึ้นติดต่อกัน 30 Epochs |
| **Image Size (imgsz)**| `640` | ขนาดของรูปภาพที่นำเข้าสู่การประมวลผล (640x640) |
| **Batch Size** | `8` | ขนาด Batch สำหรับประมวลผลต่อรอบการคำนวณ |
| **Optimizer** | `MuSGD` / `Auto` | อัลกอริทึมในการปรับแต่ง Weights |
| **Mosaic** | `1.0` | การนำภาพ 4 ภาพมารวมกันเพื่อเพิ่มความหลากหลายของมุมมอง |
| **Mixup** | `0.1` | การซ้อนทับภาพสองภาพเพื่อเพิ่มความทนทานต่อการถูกบดบัง |
| **Degrees** | `15.0` | การสุ่มหมุนภาพไม่เกิน ±15 องศา |

---

## ❓ การแก้ไขปัญหาที่พบบ่อย (Troubleshooting & FAQs)

### Q1: เกิดข้อผิดพลาด `CUDA out of memory` ตอนรัน 02-train2.py
**แนวทางแก้ไข:** ให้เปิดไฟล์ `02-train2.py` แล้วปรับลดขนาด `batch` จาก `8` เหลือ `4` หรือ `2` และตรวจสอบว่ามีโปรแกรมอื่นดึงใช้งาน VRAM อยู่หรือไม่

### Q2: กล้องไม่เปิดทำงานเมื่อรัน 05-test-camera2.py
**แนวทางแก้ไข:** ตรวจสอบว่าไม่มีโปรแกรมอื่น (เช่น Zoom, MS Teams) กำลังดึงใช้กล้องอยู่ หากยังใช้ไม่ได้ ให้ลองเปลี่ยน Index ของกล้องในสคริปต์จาก `cv2.VideoCapture(0)` เป็น `cv2.VideoCapture(1)` หรือ `2`

### Q3: สคริปต์ 01-export_dataset.py หาไฟล์ JSON ไม่เจอ
**แนวทางแก้ไข:** ตรวจสอบให้แน่ใจว่าไฟล์ที่ Export มาจาก Label Studio วางอยู่ในโฟลเดอร์ Root เดียวกับสคริปต์ และตรวจสอบว่าชื่อไฟล์ลงท้ายด้วย `.json`
