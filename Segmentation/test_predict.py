import cv2

from ultralytics import YOLO


model = YOLO("./runs/segment/train8/weights/best.pt")

results = model.predict(source="BotPers_S2_2_591.png", show=True,classes=[0])

for result in results:
    im = result.plot()
    im = cv2.resize(im,(1200,900))
    cv2.imshow("Prediction: ",im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()