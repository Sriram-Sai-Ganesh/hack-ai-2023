import cv2
import mediapipe as mp
import time
import mouse_movement
import math
import threading
import sys
from knn import knn

mouseX = -1
mouseY = -1
updated = False
action = 30


class handDetector():


    def __init__(self, mode = False, maxHands = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mouseX = -1
        self.mouseY = -1

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x* w), int(lm.y * h)
                lmlist.append([id, cx, cy, lm.z * -10])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

def main():

    videoThread = threading.Thread(target=video, args=())
    mouseThread = threading.Thread(target=mouse, args=())

    videoThread.start()
    mouseThread.start()

    mouseThread.join()
    videoThread.join()
 


def video():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    """cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)"""
    detector = handDetector()
    knnModel = knn()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)


        if len(lmlist) != 0:
            global mouseX
            global mouseY
            global updated
            global action
            
            updated = True
            mouseX = lmlist[9][1]
            mouseY = lmlist[9][2]

            # Set action based on AI model
            knnInput = lmlist[:][1:2]
            knnInput = knnInput.flatten()
            action = knnModel.predict(knnInput)
            print(action)
        else:
            updated = False

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("window", img)
        cv2.waitKey(1)



def mouse():

    while True:

        if (mouseX != -1 and mouseY != -1 and updated):
            mouse_movement.updateMouse(mouseX, mouseY) # Pointer finger tip = index 8

            """# Pinky tip = index 20
            # Ring tip = index 16
            # Middle tip = index 12
            # Index tip = index 8
            # Thumb tip = index 4
            tipDists = [math.dist(lmlist[4][1:2], lmlist[20][1:2]), math.dist(lmlist[4][1:2], lmlist[16][1:2]), math.dist(lmlist[4][1:2], lmlist[12][1:2]), math.dist(lmlist[4][1:2], lmlist[8][1:2])]
            minDists = min(tipDists)
            if minDists < 10 + 5*lmlist[4][3]:
                finger = tipDists.index(minDists) #0: pinky     1: ring     2: middle       3: index
                if finger == 0:
                    print("pinky-thumb")
                if finger == 1:
                    print("ring-thumb")
                if finger == 2:
                    print("middle-thumb")
                if finger == 3:
                    print("index-thumb")"""
        if action == 10: # Thumbs up = middle click
            mouse_movement.middleClickHold()
        elif action == 20: # Thumbs down = right click
            mouse_movement.rightClickHold()
        elif action == 30: # Open palm = no click
            mouse_movement.unclick()
        elif action == 40: # Closed fist = left click
            mouse_movement.leftClickHold()

            

if __name__ == "__main__":
    main()
