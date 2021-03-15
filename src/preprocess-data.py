import cv2
import numpy as np
import myutils as utils

path = "../training-images-raw/"
filenames = []
extension = ".jpeg"
heightImg = 640
widthImg = 480
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

filenames = utils.getImageFiles(path)

for filename in filenames:
    pathImage = str(path + filename + extension)
    imgHighRes = cv2.imread(pathImage)
    img = cv2.resize(imgHighRes, (widthImg, heightImg))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.Canny(imgBlur, 90, 90)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)

    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest, maxArea = utils.biggestContour(contours)

    if biggest.size == 0:
        print("Cannot find the biggest contour on the image")
        exit()

    biggest = utils.reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
    imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

    docGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    docBlur = cv2.GaussianBlur(docGray, (7, 7), 1)
    docCanny = cv2.Canny(docBlur, 50, 50)
    docContour = imgWarpColored.copy()
    docContours, hierarchy = cv2.findContours(docCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for index, cnt in enumerate(docContours):
        area = cv2.contourArea(cnt)
        if area > 1000 and area < 2000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if w > (2.5 * h) and w < (5.5 * h):
                cv2.drawContours(docContour, cnt, -1, (255, 0, 0), 3)
                y = y + ((h - 21) // 2)
                x = x + ((w - 72) // 2)
                croppedRect = imgWarpColored[y:y + 21, x:x + 72]
                imgTokens = filename.split("-")
                if imgTokens[2] in ["08", "09", "10", "11", "12", "13"]:
                    cv2.imwrite("../data/test/" + imgTokens[0] + "/" + filename + "-" + str(index) + ".jpg",
                                croppedRect)
                else:
                    cv2.imwrite("../data/train/" + imgTokens[0] + "/" + filename + "-" + str(index) + ".jpg",
                                croppedRect)

    imageArray = ([img, imgBlur, imgThreshold], [imgBigContour, imgWarpColored, docContour])
    labels = [["Raw Input Image", "Blurring Image", "Finding All Contours"],
              ["Finding Biggest Contour", "Warping To Fit Page", "Finding All MCQS"]]
    stackedImage = utils.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)

    cv2.waitKey(100)

    ##############################################################################################


