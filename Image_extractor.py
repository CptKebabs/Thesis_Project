import cv2

video_path = ""

cap = cv2.VideoCapture(video_path)
frame_count = 0

if not cap.isOpened():
    print(f"Failed Opening: {video_path}")
    exit()

while True:
    ret,frame = cap.read()

    if not ret:
        break

    cv2.imshow("Video",frame)

    key = cv2.waitKey(25)#wait 25ms for key press of q

    if key & 0xFF == ord("q"):# q to exit
        break
    elif key & 0xFF == ord("s"):#s to save frame
        cv2.imwrite(f"CalibrationInput/Camera_2/Save_{frame_count}.png",frame)
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()