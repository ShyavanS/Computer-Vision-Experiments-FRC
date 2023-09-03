import cv2
w = int(input("Stream Width: "))
h = int(input("Stream Height: "))
rW = int(input("Resized Width: "))
rH = int(input("Resized Height: "))
cap = cv2.VideoCapture("http://frcvision.local:1181/?action=stream")
cap.set(3, w)
cap.set(4, h)
while (cap.isOpened()):
    if cv2.waitKey(1) & 0xFF == ord('1'):
        cap = cv2.VideoCapture("http://frcvision.local:1181/?action=stream")
    if cv2.waitKey(1) & 0xFF == ord('2'):
        cap = cv2.VideoCapture("http://frcvision.local:1182/?action=stream")
    _, frame = cap.read()
    cv2.namedWindow("raw stream", cv2.WND_PROP_FULLSCREEN)
    cv2.resizeWindow("raw stream", rW, rH)
    cv2.imshow("raw stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
