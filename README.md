# 🍿 AIYolo-Snacks: ระบบตรวจจับซองขนมด้วย YOLO26

โปรเจกต์ **Object Detection** สำหรับตรวจจับและจำแนกประเภทซองขนมจากภาพ โดยใช้
**YOLO26** และ **Ultralytics**

โมเดลรองรับการตรวจจับซองขนม 3 ประเภท ได้แก่ **Lausanne, Oreo และ Voiz**
และสามารถนำโมเดลที่ผ่านการฝึกไปใช้งานกับรูปภาพ วิดีโอ และกล้องแบบ Real-time ได้

------------------------------------------------------------------------

## 📌 คุณสมบัติของระบบ

-   ตรวจจับซองขนมด้วย **Object Detection + Bounding Box**
-   จำแนกซองขนม 3 Class
-   ทดสอบโมเดลกับรูปภาพ
-   ทดสอบโมเดลกับวิดีโอ
-   ตรวจจับวัตถุผ่าน Webcam แบบ Real-time
-   แปลง Annotation จาก Label Studio JSON เป็น YOLO Dataset
-   แบ่ง Dataset เป็น Training และ Validation อัตโนมัติ
-   รองรับ Data Augmentation ระหว่างการ Training
-   มีโมเดลที่ Train แล้วอยู่ที่ `Model/best.pt`

------------------------------------------------------------------------

## 🏷️ Target Classes

Dataset ในโปรเจกต์มีทั้งหมด **3 Classes**

   Class ID  Class Name
  ---------- --------------
     `0`     **Lausanne**
     `1`     **Oreo**
     `2`     **Voiz**

รายชื่อ Class ถูกกำหนดไว้ใน:

``` text
New-Snacks_dataset/classes.txt
```

และ:

``` text
New-Snacks_dataset/data.yaml
```

ตัวอย่าง `data.yaml`:

``` yaml
train: images/train
val: images/val

nc: 3
names: ['Lausanne', 'Oreo', 'Voiz']
```

------------------------------------------------------------------------

## 📊 Dataset

Dataset ที่มีอยู่ภายในโปรเจกต์ประกอบด้วย:

-   Frame Images ทั้งหมด: **37 ภาพ**
-   Training Images: **29 ภาพ**
-   Validation Images: **8 ภาพ**
-   Train Split: **80%**
-   Validation Split: **20%**
-   Random Seed: **42**

โครงสร้าง Dataset:

``` text
New-Snacks_dataset/
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
├── classes.txt
└── data.yaml
```

------------------------------------------------------------------------

## 📁 Project Structure

``` text
AIYolo-Snacks/
│
├── Model/
│   └── best.pt
│
├── New-Snacks_dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   │
│   ├── classes.txt
│   └── data.yaml
│
├── frame/
│   └── images/
│
├── 01-export_dataset.py
├── 02-train2.py
├── 03-test_image.py
├── 04-test_video.py
├── 05-test-camera2.py
│
├── project-9-at-2026-10-05-23-47-6b8d3295.json
├── requirements.txt
├── test.jpg
└── README.md
```

> หมายเหตุ: `04-test_video.py` อ้างถึงไฟล์ `VID_20261005_230813.mp4`
> แต่ไฟล์วิดีโอนี้ไม่ได้อยู่ในชุดไฟล์โปรเจกต์ที่ส่งมานี้ หากต้องการทดสอบวิดีโอ
> ต้องนำไฟล์ดังกล่าวมาไว้ที่ Root ของโปรเจกต์ก่อน

------------------------------------------------------------------------

# 🛠️ Requirements & Installation

## 1. Python

แนะนำให้ใช้ Python เวอร์ชันที่รองรับกับ Ultralytics และ PyTorch ในระบบของผู้ใช้งาน

ตรวจสอบเวอร์ชัน Python:

``` bash
python --version
```

------------------------------------------------------------------------

## 2. สร้าง Virtual Environment

### Windows

``` bash
python -m venv env
env\Scripts\activate
```

### macOS / Linux

``` bash
python3 -m venv env
source env/bin/activate
```

------------------------------------------------------------------------

## 3. ติดตั้ง Dependencies

โปรเจกต์มีไฟล์:

``` text
requirements.txt
```

ภายในประกอบด้วย:

``` text
ultralytics
opencv-python
torch
```

ติดตั้งด้วย:

``` bash
pip install -r requirements.txt
```

หากยังไม่มี PyTorch ที่รองรับ GPU สามารถติดตั้ง PyTorch ให้เหมาะกับ Hardware
ของเครื่องก่อน แล้วจึงติดตั้ง Dependencies ที่เหลือ

------------------------------------------------------------------------

# 🔄 Project Workflow

กระบวนการทำงานของโปรเจกต์:

``` text
Frame Images
     │
     ▼
Label Studio
     │
     ▼
Export JSON
     │
     ▼
01-export_dataset.py
     │
     ▼
New-Snacks_dataset
     │
     ▼
02-train2.py
     │
     ▼
YOLO26 Model
     │
     ▼
Model/best.pt
     │
     ├──────────────┬───────────────┐
     ▼              ▼               ▼
Test Image      Test Video      Real-time Camera
03-test_image   04-test_video   05-test-camera2
```

------------------------------------------------------------------------

# 1. 📷 Data Preparation

รูปภาพต้นทางสำหรับ Dataset ถูกเก็บไว้ใน:

``` text
frame/images/
```

ในโปรเจกต์ปัจจุบันมีประมาณ **37 ภาพ**

รูปภาพเหล่านี้ถูกนำไปทำ Annotation ด้วย Label Studio ก่อนนำมาแปลงเป็น YOLO
Dataset

สำหรับ Dataset ที่มีคุณภาพ ควรมีภาพที่หลากหลาย เช่น:

-   มุมมองที่แตกต่างกัน
-   ระยะห่างจากวัตถุหลายระดับ
-   สภาพแสงที่แตกต่างกัน
-   พื้นหลังหลายรูปแบบ
-   มีวัตถุหลายชิ้นในภาพ
-   มีการบดบังวัตถุบางส่วน

------------------------------------------------------------------------

# 2. 🏷️ Annotation ด้วย Label Studio

โปรเจกต์ใช้ **Label Studio** สำหรับทำ Bounding Box Annotation

แต่ละภาพควรทำ Bounding Box รอบซองขนมที่ต้องการตรวจจับ และกำหนด Class ให้ถูกต้อง:

``` text
Lausanne
Oreo
Voiz
```

หลังจากทำ Annotation เสร็จ ให้ Export เป็น:

``` text
JSON
```

จากนั้นนำไฟล์ JSON มาไว้ใน Root Directory เดียวกับ:

``` text
01-export_dataset.py
```

ในโปรเจกต์ชุดนี้มีไฟล์:

``` text
project-9-at-2026-10-05-23-47-6b8d3295.json
```

------------------------------------------------------------------------

# 3. 🔄 Convert Label Studio → YOLO Dataset

ใช้:

``` text
01-export_dataset.py
```

สำหรับแปลงข้อมูล Annotation จาก Label Studio JSON เป็น YOLO Format

รัน:

``` bash
python 01-export_dataset.py
```

Script จะทำงานดังนี้:

1.  ค้นหาไฟล์ `.json` ใน Root Directory
2.  อ่านข้อมูล Annotation
3.  ตรวจสอบ Class ที่ถูกใช้งานจริง
4.  แปลง Bounding Box เป็น YOLO Format
5.  สุ่มข้อมูลด้วย Seed `42`
6.  แบ่งข้อมูลเป็น Train 80% และ Validation 20%
7.  Copy รูปภาพไปยัง Dataset
8.  สร้างไฟล์ Label `.txt`
9.  สร้าง `classes.txt`
10. สร้าง `data.yaml`

ค่าใน Script:

``` python
TRAIN_SPLIT = 0.8
SEED = 42
```

ผลลัพธ์:

``` text
New-Snacks_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── classes.txt
└── data.yaml
```

------------------------------------------------------------------------

# 4. 🧠 Model Training

ไฟล์ที่ใช้ Train:

``` text
02-train2.py
```

รันด้วย:

``` bash
python 02-train2.py
```

Script ใช้โมเดลเริ่มต้น:

``` text
yolo26n.pt
```

> หมายเหตุ: ไฟล์ `yolo26n.pt` ไม่ได้อยู่ในชุดไฟล์ที่ส่งมานี้ ดังนั้นหากต้องการ Train ใหม่
> ต้องเตรียมไฟล์โมเดลนี้ให้สามารถเข้าถึงได้ก่อน

Dataset ที่ใช้:

``` text
New-Snacks_dataset/data.yaml
```

------------------------------------------------------------------------

## ⚙️ Training Configuration

ค่าที่กำหนดไว้จริงใน `02-train2.py`:

  Parameter                             Value
  ------------------ ------------------------
  Base Model                     `yolo26n.pt`
  Epochs                                `150`
  Image Size                            `640`
  Optimizer                           `MuSGD`
  Device               CUDA ถ้ามี ไม่เช่นนั้นใช้ CPU
  Train Split                           `80%`
  Validation Split                      `20%`
  Degrees                               `7.0`
  Shear                                 `5.0`
  Perspective                         `0.001`
  Flip Left/Right                       `0.5`
  Flip Up/Down                          `0.5`
  Mosaic                                `0.1`
  Mixup                                 `0.1`
  Close Mosaic                           `10`

Device ถูกเลือกอัตโนมัติจาก:

``` python
device=0 if torch.cuda.is_available() else "cpu"
```

ดังนั้น:

``` text
มี NVIDIA CUDA GPU → ใช้ GPU
ไม่มี CUDA GPU      → ใช้ CPU
```

------------------------------------------------------------------------

# 5. 💾 Trained Model

โมเดลที่ผ่านการ Train แล้วถูกจัดเก็บไว้ที่:

``` text
Model/best.pt
```

ไฟล์นี้ถูกใช้โดย Script สำหรับ Inference ทั้ง 3 รูปแบบ:

``` text
03-test_image.py
04-test_video.py
05-test-camera2.py
```

------------------------------------------------------------------------

# 6. 🖼️ Test Image

ไฟล์:

``` text
03-test_image.py
```

ใช้สำหรับทดสอบโมเดลกับรูปภาพ:

``` text
test.jpg
```

รัน:

``` bash
python 03-test_image.py
```

Script จะโหลด:

``` text
Model/best.pt
```

และใช้ Confidence Threshold:

``` python
conf=0.5
```

ผลลัพธ์จะถูกแสดงผ่าน OpenCV และสามารถกดปุ่มใดก็ได้เพื่อปิดหน้าต่าง

ตัวอย่าง Workflow:

``` text
test.jpg
   ↓
Model/best.pt
   ↓
YOLO26 Detection
   ↓
Bounding Box
   ↓
Class + Confidence
```

------------------------------------------------------------------------

# 7. 🎥 Test Video

ไฟล์:

``` text
04-test_video.py
```

ใช้สำหรับตรวจจับวัตถุจากวิดีโอ

รัน:

``` bash
python 04-test_video.py
```

Script ใช้ไฟล์:

``` text
VID_20261005_230813.mp4
```

และใช้:

``` python
conf=0.5
```

ผลลัพธ์จะถูกบันทึกโดย Ultralytics ในโฟลเดอร์ประมาณ:

``` text
runs/detect/predict/
```

> ไฟล์ `VID_20261005_230813.mp4` ไม่ได้รวมอยู่ใน ZIP นี้ จึงต้องเพิ่มไฟล์เองก่อนรัน
> Script

------------------------------------------------------------------------

# 8. 📹 Real-time Camera Detection

ไฟล์:

``` text
05-test-camera2.py
```

ใช้ Webcam สำหรับตรวจจับวัตถุแบบ Real-time

รัน:

``` bash
python 05-test-camera2.py
```

ระบบจะเปิดกล้องด้วย:

``` python
cv2.VideoCapture(0)
```

และใช้ Confidence Threshold:

``` python
conf=0.25
```

Device:

``` python
device=0 if torch.cuda.is_available() else "cpu"
```

Workflow:

``` text
Webcam
   ↓
OpenCV
   ↓
YOLO26
   ↓
Object Detection
   ↓
Bounding Box + Class + Confidence
```

กด:

``` text
q
```

เพื่อออกจากระบบ

------------------------------------------------------------------------

# 📈 Detection Output

ผลการตรวจจับของ YOLO จะประกอบด้วย:

-   Bounding Box
-   Class Name
-   Confidence Score

ตัวอย่าง:

``` text
Oreo      0.92
Lausanne  0.87
Voiz      0.95
```

ค่า Confidence จะขึ้นอยู่กับภาพและผลการทำนายของโมเดลในแต่ละกรณี

------------------------------------------------------------------------

# 📂 ไฟล์สำคัญ

  File                               หน้าที่
  ---------------------------------- -----------------------------------------
  `01-export_dataset.py`             แปลง Label Studio JSON เป็น YOLO Dataset
  `02-train2.py`                     Train YOLO26
  `03-test_image.py`                 ทดสอบกับรูปภาพ
  `04-test_video.py`                 ทดสอบกับวิดีโอ
  `05-test-camera2.py`               ตรวจจับผ่าน Webcam แบบ Real-time
  `Model/best.pt`                    Trained Model สำหรับ Inference
  `New-Snacks_dataset/data.yaml`     Configuration ของ Dataset
  `New-Snacks_dataset/classes.txt`   รายชื่อ Classes
  `test.jpg`                         รูปภาพสำหรับทดสอบ
  `requirements.txt`                 Python Dependencies
  `*.json`                           Annotation Export จาก Label Studio

------------------------------------------------------------------------

# 🚀 Recommended Workflow

หากต้องการเริ่มโปรเจกต์ใหม่ตั้งแต่ต้น:

### Step 1 --- เตรียม Environment

``` bash
python -m venv env
```

เปิดใช้งาน Environment แล้วติดตั้ง:

``` bash
pip install -r requirements.txt
```

### Step 2 --- เตรียม Annotation

นำรูปภาพไปทำ Bounding Box ด้วย Label Studio

จากนั้น Export เป็น:

``` text
JSON
```

และนำ JSON มาไว้ใน Root Directory

### Step 3 --- สร้าง Dataset

``` bash
python 01-export_dataset.py
```

### Step 4 --- Train

เตรียม `yolo26n.pt` แล้วรัน:

``` bash
python 02-train2.py
```

### Step 5 --- นำ Best Model ไปใช้งาน

ตรวจสอบให้มี:

``` text
Model/best.pt
```

### Step 6 --- Test Image

``` bash
python 03-test_image.py
```

### Step 7 --- Test Video

เตรียม:

``` text
VID_20261005_230813.mp4
```

แล้วรัน:

``` bash
python 04-test_video.py
```

### Step 8 --- Real-time Camera

``` bash
python 05-test-camera2.py
```

------------------------------------------------------------------------

# ⚠️ Notes

-   Dataset ปัจจุบันมี 3 Classes: `Lausanne`, `Oreo`, `Voiz`
-   `01-export_dataset.py` ใช้ Train/Validation Split ที่ **80/20**
-   ใช้ Random Seed `42`
-   `02-train2.py` กำหนด Training จำนวน **150 Epochs**
-   Training ใช้ `MuSGD`
-   Image Size คือ `640`
-   GPU จะถูกใช้เมื่อเครื่องมี NVIDIA CUDA ที่ PyTorch ตรวจพบ
-   หากไม่มี CUDA ระบบจะใช้ CPU
-   Inference ใช้ `Model/best.pt`
-   Test Image ใช้ Confidence `0.5`
-   Test Video ใช้ Confidence `0.5`
-   Real-time Camera ใช้ Confidence `0.25`
-   ต้องมี `yolo26n.pt` หากต้องการ Train ใหม่
-   ไฟล์วิดีโอ `VID_20261005_230813.mp4` ไม่ได้รวมอยู่ใน ZIP ที่ส่งมา
-   Dataset และโมเดลที่มีอยู่ในโปรเจกต์สามารถใช้สำหรับการทดสอบได้ทันทีตามไฟล์ที่มี

------------------------------------------------------------------------

# 📌 Project Summary

**AIYolo-Snacks** เป็นโปรเจกต์ Computer Vision
สำหรับตรวจจับและจำแนกซองขนมด้วย **YOLO26**

ระบบสามารถตรวจจับ:

``` text
┌─────────────────────────┐
│       AIYolo-Snacks     │
└────────────┬────────────┘
             │
             ▼
      ┌──────────────┐
      │    YOLO26    │
      └──────┬───────┘
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
  Lausanne  Oreo     Voiz
     │       │        │
     └───────┼────────┘
             ▼
     Bounding Box
     + Class
     + Confidence
```

รองรับการใช้งาน 3 รูปแบบ:

``` text
Image
  ↓
03-test_image.py

Video
  ↓
04-test_video.py

Webcam
  ↓
05-test-camera2.py
```

โมเดลที่ผ่านการ Train แล้ว:

``` text
Model/best.pt
```

จึงสามารถนำไปใช้สำหรับตรวจจับซองขนมจาก **รูปภาพ วิดีโอ และกล้องแบบ Real-time**
ได้
