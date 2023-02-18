import pyautogui

def updateMouse(x,y):

    #print(str(x) + " " + str(y))
    pyautogui.moveTo(1920-2*x, y*1.5, duration = 0)
