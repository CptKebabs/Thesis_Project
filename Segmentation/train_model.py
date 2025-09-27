from ultralytics import YOLO

#from https://docs.ultralytics.com/modes/train/

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

results = model.train(data="seaweed-seg.yaml", epochs=100, imgsz=640)#etc more to be added