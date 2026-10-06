import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


def main():
    model_path = Path(__file__).resolve().parent / "Model" / "best.pt"
    model = YOLO(str(model_path))

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    print("กด 'q' เพื่อออกจากโปรแกรม")

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(frame, device=device, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO26 Real-time Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()