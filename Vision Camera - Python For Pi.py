import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from networktables import NetworkTables


def canny(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(grey, 13)
    canny = cv2.Canny(blur, 350, 650)
    return canny


def display_lines(img, lines):
    line_image = np.zeros_like(img)
    table = NetworkTables.getTable('SmartDashboard')
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            rise = y2-y1
            run = x2-x1
            m = rise/run
            angle = math.degrees(math.atan(m))
            centerX = x1+(run/2)
            centerY = y1+(rise/2)
            if angle < 0:
                angle += 180
            if angle >= 70 and angle <= 110:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                table.putNumber('CenterX', centerX)
                table.putNumber('CenterY', centerY)
                table.putNumber('Angle', angle)
    return line_image


cap = cv2.VideoCapture("http://frcvision.local:1181/?action=stream")
cap.set(3, 640)
cap.set(4, 480)
NetworkTables.initialize(server='10.50.32.2')
while (cap.isOpened()):
    _, frame = cap.read()
    canny_img = canny(frame)
    line_object = cv2.HoughLinesP(
        canny_img, 2, np.pi/180, 100, np.array([]), minLineLength=100, maxLineGap=5)
    line_img = display_lines(canny_img, line_object)
    combo_img = cv2.addWeighted(canny_img, 0.8, line_img, 1, 1)
    cv2.imshow("processed stream", combo_img)  # for debugging
    if cv2.waitKey(1) & 0xFF == ord('q'):  # for debugging
        break  # for debugging
cap.release()  # for debugging
cv2.destroyAllWindows()  # for debugging
