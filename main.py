from cvzone.ClassificationModule import Classifier
import cv2
import numpy as np
import extract_pokemon
import overlay_pokemon


#method for scanning in Tepig
def chooseTepig(insert = False):
    cap = cv2.VideoCapture(0)
    maskClassifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
    cap.set(3, 1920)
    cap.set(4, 1080)

    while True:
        _, img = cap.read()

        #cv2.normalize(img, img, 50, 255, cv2.NORM_MINMAX)
        # cv2.rectangle(img, (480, 170), (960, 80), color=(0, 225, 0), thickness=2)
        prediction = maskClassifier.getPrediction(img)
        # print(prediction)

        if insert:
            posX = 200
            posY = 200
            img = overlay_pokemon.overlay_pokemon(img, posX, posY, prediction[1])   # prediction[1] is argmax value

        # display cam, press q to exit
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Method for scanning in Bulbasaur
def chooseBulbasaur():
    cap = cv2.VideoCapture(0)
    maskClassifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
    cap.set(3, 1920)
    cap.set(4, 1080)


    while True:
        _, img = cap.read()
        # cv2.normalize(img, img, 50, 255, cv2.NORM_MINMAX)
        cv2.rectangle(img, (480, 170), (960, 80), color=(0, 225, 0), thickness=2)
        prediction = maskClassifier.getPrediction(img)
        print(prediction)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

#Method for scanning in totodile
def chooseTotodile():
    cap = cv2.VideoCapture(0)
    maskClassifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
    cap.set(3, 1920)
    cap.set(4, 1080)

    while True:
        _, img = cap.read()
        # cv2.normalize(img, img, 50, 255, cv2.NORM_MINMAX)
        cv2.rectangle(img, (480, 170), (960, 80), color=(0, 225, 0), thickness=2)
        prediction = maskClassifier.getPrediction(img)
        print(prediction)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    #  # press any key to exit preview
    # extract_pokemon.test_extract()
    # extract_pokemon.test_segment()
    chooseTepig(insert = True)