import cv2
import numpy as np

video_path_top = "./[Electro House] Virtual Riot - Energy Drink-360p.mp4"#top video
video_path_bot = "./Tristam - I Remember [Monstercat Release]-360p.mp4"#bottom video

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

    #note if we run into size difference problems we can use cv2.resize
    
    if not ret1 or not ret2:
        break
    final_frame = np.concatenate((frame1,frame2),axis=1)

    cv2.imshow("Video",final_frame)
    #cv2.imshow("Video",frame2)

    key = cv2.waitKey(25)#wait 25ms for key press of q

    if key & 0xFF == ord("q"):# q to exit
        break
    elif key & 0xFF == ord("s"):#s to save frames
        cv2.imwrite(f"ImageExtractorOutput/TopPers_{frame_count}.png",frame1)
        cv2.imwrite(f"ImageExtractorOutput/BotPers_{frame_count}.png",frame2)
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()