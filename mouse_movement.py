import pyautogui



def updateMouse(x,y):

    pyautogui.moveTo(1920-2*x, y*1.5, duration = .1)

def leftClickHold():
    pyautogui.mouseDown(button='left')

def middleClickHold():
    pyautogui.mouseDown(button='middle')

def rightClickHold():
    pyautogui.mouseDown(button='right')

def unclick():
    pyautogui.mouseUp(button='left')