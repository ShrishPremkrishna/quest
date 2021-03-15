import cv2
import csv
import numpy as np
import myutils as utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import emoji
from PIL import Image, ImageFont, ImageDraw

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

assignmentImgsPath = "../assessment-images-raw/"
assignmentKeysPath = "../assessment-keys/"
filenames = utils.getImageFiles(assignmentImgsPath)
extension = ".JPEG"
heightImg = 640
widthImg = 480
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
mcqRectFolderPath = "../assessment-mcqs/"

modelFilename = "Control-10_Epochs.h5"
mcqPredictionModel = load_model(modelFilename)

# Iterate each file (assignment submission) in directory path
for filename in filenames:

    # Get the name of the image file without extension
    filename = filename.split(".")[0]
    pathImage = str(assignmentImgsPath + filename + extension)
    img = cv2.imread(pathImage)
    print(pathImage)

    # Get answer key
    assessmentId = filename.split("-")[0]
    with open(assignmentKeysPath + assessmentId + '.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    [assessmentKeys] = data

    # Resize the image to a low resolution image
    height, width, channels = img.shape
    img = cv2.resize(img, (widthImg, heightImg))

    # Use OpenCV library to find contours on the image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.Canny(imgBlur, 90, 90)

    # Find the biggest contour
    # Biggest contour on the image file will be the assignment sheet
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, maxArea = utils.biggestContour(contours)

    # If picture is not taken correctly, capturing all the edges of the assignment sheet
    if biggest.size == 0:
        print("Cannot find the biggest contour on the image")
        continue



# If the assignment sheet is warped on the image, fix it.
    biggest = utils.reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
    imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    # Find contours on the assignment sheet
    docGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    docBlur = cv2.GaussianBlur(docGray, (7,7), 1)
    docCanny = cv2.Canny(docBlur, 50, 50)
    docContour = imgWarpColored.copy()
    docContours, hierarchy = cv2.findContours(docCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Identify the MCQ rectangles from the contours
    mcqRects = []
    for index, cnt in enumerate(docContours):
        area = cv2.contourArea(cnt)
        if area > 500 and area < 4000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if (2.5 * h) < w < (5.5 * h):
                cv2.drawContours(docContour, cnt, -1, (255, 255, 0), 3)
                mcqRects.append([x, y, w, h])

    # Sort the MCQ rectangles based on their Y values
    mcqRects.sort(key=lambda rect: rect[1])
    docPredictions = docContour.copy()

    for index, mcqRect in enumerate(mcqRects):
        [x, y, w, h] = mcqRect
        y = y + ((h - 21) // 2)
        x = x + ((w - 72) // 2)

        # Crop and store the MCQ rectangles each into separate files
        mcqImage = imgWarpColored[y:y + 21, x:x + 72]
        mcqFilePath = mcqRectFolderPath + filename + "-" + str(index + 1) + ".jpg"
        cv2.imwrite(mcqFilePath, mcqImage)

        # Prep MCQ image for model prediction
        image_shape = mcqImage.shape
        mcqImage = image.img_to_array(mcqImage)
        mcqImage = np.expand_dims(mcqImage, axis=0)

        # Make predictions
        predictions = mcqPredictionModel.predict_classes(mcqImage)
        predictedAnswer = 'N'
        if predictions[0] == 0:
            predictedAnswer = 'A'
        elif predictions[0] == 1:
            predictedAnswer = 'B'
        elif predictions[0] == 2:
            predictedAnswer = 'C'
        elif predictions[0] == 3:
            predictedAnswer = 'D'
        mcqRect.append(predictedAnswer)
        mcqRect.append(assessmentKeys[index])
        mcqRect.append(predictedAnswer == assessmentKeys[index])
        cv2.putText(docPredictions, predictedAnswer, (x + 85, y + 15),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
        print("predictions", predictions)

    # Write evaluations on assessment sheet
    docEvaluation = Image.fromarray(docPredictions)
    draw = ImageDraw.Draw(docEvaluation)
    font = ImageFont.truetype("Arial Unicode.ttf", 32)
    correctAnswerCount = 0
    for index, mcqRect in enumerate(mcqRects):
        if mcqRect[6]:
            correctAnswerCount = correctAnswerCount + 1
            draw.text((mcqRect[0] + 105, mcqRect[1] - 20), "\u2713", (0, 255, 0), font=font)
        else:
            draw.text((mcqRect[0] + 105, mcqRect[1] - 20), "\u2717", (0, 0, 255), font=font)

    # Write score on the assessment sheet
    docScore = np.array(docEvaluation).copy()
    score = (correctAnswerCount * 100) // len(mcqRects)
    cv2.putText(docScore, str(score) + "%", (300, 100),
                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

    # Show the transformation as stacked images
    imageArray = (
        [img, imgThreshold, imgBigContour, imgWarpColored],
        [docContour, docPredictions, np.array(docEvaluation), docScore]
    )
    labels = [
        ["Original Image", "Find All Contours", "Find Largest Contour", "Warp Assessment Sheet"],
        ["Find MCQs", "Make Predictions", "Evaluate Answers", "Score Assignment"]
    ]
    stackedImage = utils.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)
    cv2.waitKey(0)