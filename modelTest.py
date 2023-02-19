import xgboost as xgb

import cv2
import mediapipe as mp
import numpy as np

class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        selfdetectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                                        max_num_hands=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist


def normalize(coordinateList):
    x9 = coordinateList[9][1]
    y9 = coordinateList[9][2]

    # Normalize the values
    coord_list = []
    for subList in coordinateList:
        coord_list.append(subList[1]-x9)
        coord_list.append(subList[2]-y9)

    return coord_list

def classify(data):
    hand_classifer = xgb.XGBClassifier()
    hand_classifer.load_model('basicModel.json')
    data = data.reshape((1,42))
    prediction = hand_classifer.predict(data)
    predToText = {0:"Thumbs Up", 1: "Thumbs Down", 2:"Open Palm", 3:"Closed Fist"}
    print(predToText.get(prediction[0]))


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    detector = handDetector()

    x = 0
    while x<1000:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            coord_list = normalize(lmlist)
            classify(np.array(coord_list))
        cv2.imshow("window", img)
        cv2.waitKey(1)
        x += 1

if __name__ == "__main__":
    main()