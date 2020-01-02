import pandas as pd
## Baseline3
def setlabel(x):
    if x<=0.01:
        return 0
    if x>0.01:
        return 1

def changecodetoint(x):
    return int(x[2:])

from keras.utils import to_categorical
newpdData1=pd.read_csv("./OriginalDataConcat.csv")
targetcolumns=["Code","Date","Open","High","Low","Close","Gain","Volumns","MA_5","MA_10","MA_20"]
newlist=newpdData1.loc[:,targetcolumns]
newlist.head()
newpdData1=newlist
newpdData1["label"]=newpdData1["Gain"].apply(setlabel)
newpdData1.fillna(0,inplace=True)
newpdData1.loc[:,"Date"]=pd.to_datetime(newpdData1.loc[:,"Date"])
newpdData1.sort_values(by=["Code","Date"],ascending=True,inplace=True)
newpdData1=newpdData1.set_index("Date",drop=True)
newpdData1["Code"]=newpdData1["Code"].apply(changecodetoint)
newpdData1.head()
newpdData1.to_csv("./2_output_Simple_0.01.csv")

