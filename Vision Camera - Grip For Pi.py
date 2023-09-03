import cv2
import numpy
import matplotlib.pyplot as plt
import math
from networktables import NetworkTables
from enum import Enum


class pipeline:
    @staticmethod
    def desaturate(src):
        """Converts a color image into shades of gray.
        Args:
            src: A color numpy.ndarray.
        Returns:
            A gray scale numpy.ndarray.
        """
        (a, b, channels) = src.shape
        if (channels == 1):
            return numpy.copy(src)
        elif (channels == 3):
            return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        elif (channels == 4):
            return cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
        else:
            raise Exception("Input to desaturate must have 1, 3 or 4 channels")

    @staticmethod
    def blur(src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The blurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        if (type is BlurType.Box_Blur):
            ksize = int(2 * round(radius) + 1)
            return cv2.blur(src, (ksize, ksize))
        elif (type is BlurType.Gaussian_Blur):
            ksize = int(6 * round(radius) + 1)
            return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
        elif (type is BlurType.Median_Filter):
            ksize = int(2 * round(radius) + 1)
            return cv2.medianBlur(src, ksize)
        else:
            return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def canny(image, thres1, thres2, aperture_size, gradient):
        """Applies a canny edge detection to the image.
        Args:
           image: A numpy.ndarray as the input.
           thres1: First threshold for the canny algorithm. (number)
           thres2: Second threshold for the canny algorithm. (number)
           aperture_size: Aperture size for the canny operation. (number)
           gradient: If the L2 norm should be used. (boolean)
        Returns:
            The edges as a numpy.ndarray.
        """
        return cv2.Canny(image, thres1, thres2, apertureSize=(int)(aperture_size),
                         L2gradient=gradient)


class line:

    def init(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def centerx(self):
        return self.x1 + ((self.x2 - self.x1)/2)

    def length(self):
        return numpy.sqrt(pow(self.x2 - self.x1, 2) + pow(self.y2 - self.y1, 2))

    def angle(self):
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))


@staticmethod
def find_lines(input):
    """Finds all line segments in an image.
    Args:
        input: A numpy.ndarray.
    Returns:
        A filtered list of Lines.
    """
    detector = cv2.createLineSegmentDetector()
    if (len(input.shape) == 2 or input.shape[2] == 1):
        lines = detector.detect(input)
    else:
        tmp = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        lines = detector.detect(tmp)
    output = []
    if len(lines) != 0:
        for i in range(1, len(lines[0])):
            tmp = Line(lines[0][i, 0][0], lines[0][i, 0][1],
                       lines[0][i, 0][2], lines[0][i, 0][3])
            output.append(tmp)
    return output


@staticmethod
def filter_lines(inputs, min_length, angle):
    """Filters out lines that do not meet certain criteria.
    Args:
        inputs: A list of Lines.
        min_Lenght: The minimum lenght that will be kept.
        angle: The minimum and maximum angles in degrees as a list of two numbers.
    Returns:
        A filtered list of Lines.
    """
    table = NetworkTables.getTable('SmartDashboard')
    outputs = []
    for line in inputs:
        if (line.length() > min_length):
            if ((line.angle() >= angle[0] and line.angle() <= angle[1]) or
                    (line.angle() + 180.0 >= angle[0] and line.angle() + 180.0 <= angle[1])):
                outputs.append(line)
                table.putNumber('CenterX', line.centerx())
                table.putNumber('Angle', line.angle())
    return outputs


BlurType = Enum(
    'BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')
cap = cv2.VideoCapture("http://frcvision.local:1181/?action=stream")
cap.set(3, 640)
cap.set(4, 480)
NetworkTables.initialize(server='10.50.32.2')
while (cap.isOpened()):
    _, frame = cap.read()
    grey_img = pipeline.desaturate(frame)
    blur_img = pipeline.blur(
        grey_img, BlurType.Median_Filter, 6.306306306306306)
    canny_img = pipeline.canny(blur_img, 350.0, 650.0, 3.0, False)
    line_img = line.find_lines(canny_img)
    filter_img = line.filter_lines(
        line_img, 40.0, [207.19424460431654, 293.33333333333337])
    cv2.imshow("processed stream", filter_img)  # for debugging
    if cv2.waitKey(1) & 0xFF == ord('q'):  # for debugging
        break  # for debugging
cap.release()  # for debugging
cv2.destroyAllWindows()  # for debugging
