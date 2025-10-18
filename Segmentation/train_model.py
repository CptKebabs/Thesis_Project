from ultralytics import YOLO
import torch

#from https://docs.ultralytics.com/modes/train/

if __name__ == '__main__':

    print(f"CUDA availiablity: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

    try:
        results = model.train(data="seaweed-seg.yaml", epochs=100, imgsz=640, batch=4, device="cpu", workers=0)
    #device 0 is gpu 0 and imagesize 640 and batch together should be considered for vram limits, workers is subprocesses active (processes with own memory spaces)
    except Exception as e:
        print(f"Training Stopped due to exception: {e}")

    print("Training Process Completed")