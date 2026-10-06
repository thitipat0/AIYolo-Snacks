# 🍿 AIYolo-Snacks: ตรวจจับและจำแนกซองขนมด้วย YOLO26

โปรเจกต์ระบบวิเคราะห์ภาพเพื่อตรวจจับและจำแนกชนิดของซองขนม 3 ยี่ห้อ (**Oreo**, **Lausanne**, **Voiz**) แบบอัตโนมัติ โดยใช้สถาปัตยกรรม **YOLO26** ร่วมกับ **Label Studio** ในการจัดการและเตรียมชุดข้อมูล สามารถประมวลผลได้ทั้งจากภาพถ่าย วิดีโอ และกล้องเว็บแคมแบบ Real-time

## 🚀 Quick Start (เริ่มต้นใช้งานด่วน)

หากคุณมีไฟล์โมเดลที่เทรนสำเร็จแล้วและต้องการทดสอบใช้งานทันที:

1. ทำตามขั้นตอนการติดตั้งในหัวข้อ [Installation](#-installation)

2. วางไฟล์โมเดลไว้ที่ตำแหน่ง `Model/best.pt`

3. รันสคริปต์เพื่อทดสอบใช้งาน:

   * **ทดสอบผ่านกล้อง Real-time:**

     ```bash
     python 05-test-camera2.py
     ```

   * **ทดสอบกับรูปภาพ:**

     ```bash
     python 03-test_image.py
     ```

## 📁 Project Structure (โครงสร้างโปรเจกต์)

```text
AIYolo-Snacks/
├── Model/
│   └── best.pt                           # ไฟล์ Weight ของโมเดลที่เทรนเสร็จแล้ว
├── New-Snacks_dataset/                   # ชุดข้อมูลรูปภาพและ Label ในรูปแบบ YOLO Dataset
├── frame/                                # โฟลเดอร์เก็บรูปภาพเฟรมที่รวบรวมไว้สำหรับ Label Studio
├── 01-export_dataset.py                  # สคริปต์แปลง Annotation จาก Label Studio เป็น YOLO Format
├── 02-train2.py                          # สคริปต์สำหรับสั่งการสั่งเทรนโมเดล (Training)
├── 03-test_image.py                      # สคริปต์สำหรับทดสอบการตรวจจับกับไฟล์รูปภาพ
├── 04-test_video.py                      # สคริปต์สำหรับทดสอบการตรวจจับกับไฟล์วิดีโอ
├── 05-test-camera2.py                    # สคริปต์ทดสอบการตรวจจับ Real-time ผ่านกล้องเว็บแคม
├── project-9-at-2026-10-05-23-47-6...json # ไฟล์ Export ข้อมูล Annotation จาก Label Studio
├── requirements.txt                      # รายการไลบรารีที่ต้องใช้ในโปรเจกต์
├── test.jpg                              # ตัวอย่างรูปภาพสำหรับทดสอบระบบ
└── README.md                             # คู่มือและเอกสารอธิบายโปรเจกต์
```

### 🏷 Target Classes

โมเดลรองรับการจำแนกขนมทั้งหมด 3 คลาส ได้แก่:

* `0`: **Oreo**
* `1`: **Lausanne**
* `2`: **Voiz**

## 🛠️ Installation (การติดตั้งระบบ)

### 1. สร้างและเปิดใช้งาน Virtual Environment

```bash
# สร้าง virtual environment ชื่อ env
python -m venv env

# เปิดใช้งาน (Windows Command Prompt)
env\Scripts\activate

# เปิดใช้งาน (Windows PowerShell)
.\env\Scripts\Activate.ps1
```

### 2. ติดตั้ง PyTorch รองรับ GPU (CUDA)

ตรวจสอบเวอร์ชัน CUDA ของเครื่องท่าน และเลือกติดตั้งตามคำสั่งที่เหมาะสม (ตัวอย่างสำหรับ CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. ติดตั้ง Dependencies ทั้งหมด

```bash
pip install -r requirements.txt
```

## 🔄 Workflow Pipeline (ขั้นตอนการทำงาน)

```text
[ถ่ายภาพ/เตรียมรูปภาพ] --> [Label Studio] --> [Export JSON] --> [01-export_dataset.py]
                                                                        │
[Testing Scripts] <-- [Model/best.pt] <-- [02-train2.py] <-- [New-Snacks_dataset]
```

### Step 1: ติดตั้งและตั้งค่า Label Studio

1. ติดตั้งและเริ่มทำงาน Label Studio
2. ตั้งค่า **Labeling Interface** โดยใช้ Template XML ด้านล่างนี้:

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

3. ทำการกำกับภาพ (Annotate) รูปภาพทั้งหมดในโฟลเดอร์ `frame/`
4. เมื่อเสร็จสิ้น ให้ทำการ Export ข้อมูลออกมาเป็นไฟล์รูปแบบ **JSON** และนำไฟล์มาวางไว้ที่ Root Directory ของโปรเจกต์ (เช่น `project-9-at-2026-10-05-23-47-6...json`)

### Step 2: แปลงชุดข้อมูลเข้าสู่ YOLO Format

รันสคริปต์เพื่อนำไฟล์ JSON มาจัดสัดส่วนและแปลงเป็น YOLO Dataset ไว้ในโฟลเดอร์ `New-Snacks_dataset/`:

```bash
python 01-export_dataset.py
```

### Step 3: การเทรนโมเดล (Training)

สั่งเริ่มการเทรนโมเดลด้วย YOLO26 โดยรันสคริปต์:

```bash
python 02-train2.py
```

*หมายเหตุ: เมื่อเทรนเสร็จสมบูรณ์ ให้นำไฟล์น้ำหนักที่ได้ผลลัพธ์ดีที่สุด (`best.pt`) มาจัดเก็บไว้ที่โฟลเดอร์ `Model/best.pt`*

### Step 4: การทดสอบประมวลผล (Testing)

* **ทดสอบจำแนกภาพแบบ Real-time จากกล้อง:**

  ```bash
  python 05-test-camera2.py
  ```

* **ทดสอบจำแนกภาพถ่าย:**

  ```bash
  python 03-test_image.py
  ```

* **ทดสอบจำแนกวิดีโอ:**

  ```bash
  python 04-test_video.py
  ```

## 💡 Troubleshooting (การแก้ไขปัญหาเบื้องต้น)

* **ปัญหา CUDA ไม่ทำงาน / GPU OOM (Out of Memory):**
  หากเกิดข้อผิดพลาดด้านหน่วยความจำ GPU เต็ม ให้ทดลองลดขนาด `batch` ในไฟล์ `02-train2.py` หรือตรวจสอบการติดตั้ง CUDA PyTorch ด้วยคำสั่ง `python -c "import torch; print(torch.cuda.is_available())"`

* **หาไฟล์ JSON ไม่พบในขั้นตอน Export:**
  ตรวจสอบว่าไฟล์ Export จาก Label Studio นำมาวางไว้ที่ Root directory ของโปรเจกต์แล้วเรียบร้อยหรือไม่
