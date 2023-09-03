import cv2
cap = cv2.VideoCapture("http://frcvision.local:1181/?action=stream")
while (cap.isOpened()):
    _, frame = cap.read()
    cv2.namedWindow("raw stream", cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow("raw stream", 640, 480)
    cv2.imshow("raw stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
