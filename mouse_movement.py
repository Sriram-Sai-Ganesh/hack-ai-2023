import pyautogui

def updateMouse(x,y):
    #print(str(x) + " " + str(y))
    pyautogui.moveTo(1080 - x, y, duration = 0)
