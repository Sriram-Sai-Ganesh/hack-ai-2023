import pyautogui

def updateMouse(x,y):


    print(x)
    #print(str(x) + " " + str(y))
    pyautogui.moveTo(1920-2*x, y, duration = 0)
