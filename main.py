from cvzone.ClassificationModule import Classifier
import cv2




#method for scanning in Tepig
def chooseTepig():
    cap = cv2.VideoCapture(0)
    maskClassifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
    cap.set(3, 1920)
    cap.set(4, 1080)

    while True:
        _, img = cap.read()

        #cv2.normalize(img, img, 50, 255, cv2.NORM_MINMAX)
        cv2.rectangle(img, (480, 170), (960, 80), color=(0, 225, 0), thickness=2)
        prediction = maskClassifier.getPrediction(img)
        print(prediction)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

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



    chooseTepig()