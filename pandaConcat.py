import pandas as pd

closedFist = pd.read_csv('testClosedFist.csv')
openFist = pd.read_csv('testOpenPalm.csv')
thumbD = pd.read_csv('thumbsdown.csv')
thumbU = pd.read_csv('thumbsup.csv')

allData = pd.concat([closedFist, openFist, thumbD, thumbU])
allData.to_csv('completeData.csv')