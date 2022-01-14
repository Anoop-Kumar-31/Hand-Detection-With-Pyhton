from mediapipe.python.solutions.hands import (Hands as _genHandsCLS,HAND_CONNECTIONS,HandLandmark,)
from mediapipe.python.solutions.drawing_utils import draw_landmarks
import cv2
from time import time

def drawFPS(img, pTime: float) -> float:
    cTime = time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime

    cv2.putText(img, f"FPS: {fps}", (10, 450), fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=3, color=(255, 0, 0), fontScale=3)
    return pTime


class HandDetector:
    def __init__(self,mode=False,maxHands=2,modelComplexity=1,minDetectionCon=0.6,minTrackingCon=0.6):
        self.hands = _genHandsCLS(mode,maxHands,modelComplexity,minDetectionCon,minTrackingCon,)

    def findHands(self, img, draw=True, HAND_CONNECTIONS=HAND_CONNECTIONS):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    draw_landmarks(img, handLms, HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0, draw=True):
        """Return the complete list of landmarks IF a hand was detected
        Else return empty list"""
        lmList = []
        # This will check if any hands have been detected
        if self.results.multi_hand_landmarks:
            # Selects the first hand detected by mediapipe
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                # finding coordinates of the landmark on the screen
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.putText(img,f"{id}",(cx,cy),fontFace=cv2.FONT_HERSHEY_PLAIN,thickness=1,color=(255, 0, 0),fontScale=1)
        return lmList

# FOR TESTING
def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    pTime = 0

    while True:
        _, img = cap.read()
        img = detector.findHands(img)
        detector.findPosition(img)

        # pTime = drawFPS(img, pTime)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
