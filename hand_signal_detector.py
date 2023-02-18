import cv2
import mediapipe as mp
import pandas
from itertools import cycle

class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
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


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    detector = handDetector()

    # thumbs up is 10
    # thumbs down is 20
    # open palm is 30
    # closed palm fist is 40

    hand_signal = 10

    col_name = []
    for i in range(21):
        col_name.append(i)
        col_name.append(i)
    col_name = [str(x) for x in col_name]
    coord_labels = ["x", "y"]
    col_name = zip(cycle(coord_labels), col_name)
    labels = []
    for blah in col_name:
        labels.append(''.join(blah[0] + blah[1]))
    # Create a list with the first label as the desired signal and the rest of the labels as indices 0-20 (inclusive)
    hand_loc_data = pandas.DataFrame(columns=["Hand_Signal"]+labels)

    x = 0
    while x<1000:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            coord_list = []
            for subList in lmlist:
                coord_list.append(subList[1])
                coord_list.append(subList[2])
            coord_list = [hand_signal] + coord_list
            print(coord_list)
            hand_loc_data.loc[len(hand_loc_data)] = coord_list


        cv2.imshow("window", img)
        cv2.waitKey(1)
        x += 1

    hand_loc_data.to_csv('test.csv')


if __name__ == "__main__":
    main()