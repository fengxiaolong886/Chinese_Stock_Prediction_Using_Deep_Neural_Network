
## Baseline1 Data process
def setlabel(x):
    if x<=-0.07:
        return 0
    if x>-0.07 and x<=-0.05:
        return 1
    if x>-0.05 and x<=-0.02:
        return 2
    if x>-0.02 and x<=0:
        return 3
    if x>0 and x<=0.02:
        return 4
    if x>0.02 and x<==0.05:
        return 5
    if x>0.05 and x<=0.07:
        return 6
     if x>0.07:
         return 7
         
from keras.utils import to_categorical
newpdData1=pd.read_csv("./OriginalDataConcat.csv")
targetcolumns=["Code","Date","Open","High","Low","Close","Gain","Volumns","MA_5","MA_10","MA_20"]
newlist=newpdData1.loc[:,targetcolumns]
newlist.head()
newpdData1=newlist
newpdData1["label"]=newpdData1["Gain"].apply(setlabel)
newpdData1.fillna(0,inplace=True)
newpdData1.loc[:,"Date"]=pd.to_datetime(newpdData1.loc[:,"Date"])
newpdData1.head()
newpdData1.sort_values(by=["Code","Date"],ascending=True,inplace=True)
newpdData1=newpdData1.set_index("Date",drop=True)
def changecodetoint(x):
    return int(x[2:])
newpdData1["Code"]=newpdData1["Code"].apply(changecodetoint)
newpdData1.head()
newpdData1.to_csv("./SimpleVersionData.csv")
data=np.array(newpdData1)


## Baseline2
def setlabel(x):
    if x<=-0.02:
        return 0
    if x>--0.02 and x<=0:
        return 1
    if x>0 and x<=0.02:
        return 2
    if x>0.02:
        return 3

def changecodetoint(x):
    return int(x[2:])

from keras.utils import to_ctegorical
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
newpdData1.to_csv("./4_output_Simple.csv")
data=np.array(newpdData1)

## Baseline3
def setlabel(x):
    if x<=0.01:
        return 0
    if x>0.01:
        return 1

def changecodetoint(x):
    return int(x[2:])

from keras.utils import to_ctegorical
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
data=np.array(newpdData1)


#Version3 Data Processing
filename=os.listdir("./data/")
pddata=pd.DataFrame()
for eachfile in filename:
    filepath="./data/"
    testfile=os.path.join(filepath,eachfile)
    try:
        test=pd.read_csv(testfile,encoding="gb2312")
        newcolumn=["Code","Name","Date","Industury","Concept","Region","Open","High"
            "Low","Close","PostRecovery","PreRecovery","Gain","Volumn","Trunover","HandTurnoverRate",
            "CirculationMarketValue","Capitalization","Toplimit","BottomLimit","PETTM","MSTTM","MRTTM",
            "PBRatio","MA_5","MA_10","MA_20","MA_30","MA_60","MA_GoldFork",
            "MACD_DIF","MACD_DEA","MACD_MACD","MACD_GoldFolk","KDJ_K",
            "KDJ_D","KDJ_J","KDJ_GoldFork","BollingerMid","BollingerUP","BollingerDown","psy",
            "psyma","rsi1","rsi2","rsi3","Amplitude","VolumnRatio"]
        test.columns=newcolumn
        del test["Name"]
        del test["Code"]
        del test["Industury"]
        del test["Concept"]
        del test["Region"]
        del test["MA_GoldFork"]
        del test["MACD_GoldFolk"]
        del test["KDJ_GoldFork"]
        slicedata=test
        slicedata.loc[:,"Date"]=pd.to_datetime(slicedata.loc[:,"Date"])
        slicedata=slicedata.set_index("Date",drop=True)
        slicedata=slicedata.sort_index(ascending=False)
        slicedata["label"]=slicedata.Gain>0
        slicedata.label.replace(True,1,inplace=True)
        slicedata.fillna(0,inplace=True)
        pdData=pd.concat([pdData,slicedata],axis=0)
    except:
        print testfile
        continue

pdData.to_csv("AllDataPandas.csv")
data=np.array(pdData)
