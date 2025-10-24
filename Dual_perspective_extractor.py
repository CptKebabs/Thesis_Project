import cv2
import numpy as np

aspect_4_3 = True #doing this because iMovie which i used to trim my videos forces a 16:9 aspect ratio

video_path_top = ""#top video
video_path_bot = ""#bottom video

cap = cv2.VideoCapture(video_path_top)
cap2 = cv2.VideoCapture(video_path_bot)
frame_count = 0

if not cap.isOpened():
    print(f"Failed Opening: {video_path_top}")
    exit()

if not cap2.isOpened():
    print(f"Failed Opening: {video_path_bot}")
    exit()

while True:
    ret1,frame1 = cap.read()
    ret2,frame2 = cap2.read()

    resized_frame1 = cv2.resize(frame1,(600,600),interpolation = cv2.INTER_NEAREST)
    resized_frame2 = cv2.resize(frame2,(600,600),interpolation = cv2.INTER_NEAREST)

    #note if we run into size difference problems we can use cv2.resize
    
    if not ret1 or not ret2:
        break
    final_frame = np.concatenate((resized_frame1,resized_frame2),axis=1)

    cv2.imshow("Video",final_frame)
    #cv2.imshow("Video",frame2)

    if aspect_4_3:
        frame1 = frame1[:, 480:-480, :]#then crop the black spots
        frame2 = frame2[:, 480:-480, :]

    key = cv2.waitKey(25)#wait 25ms for key press of q

    if key & 0xFF == ord("q"):# q to exit
        break
    elif key & 0xFF == ord("s"):#s to save frames
        cv2.imwrite(f"ImageExtractorOutput/Test Pairs/TopPers{frame_count}.png",frame1)
        cv2.imwrite(f"ImageExtractorOutput/Test Pairs/BotPers{frame_count}.png",frame2)
    elif key & 0xFF == ord("p"):#p to pause
        cv2.waitKey(0)# wait until next keypress
    
    frame_count += 1

cap.release()
cap2.release()
cv2.destroyAllWindows()