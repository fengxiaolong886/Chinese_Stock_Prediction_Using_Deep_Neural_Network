#!/usr/bin/env python
# coding: utf-8

# # 1. 导入相关库文件

# In[1]:


import os
import pandas as pd
import numpy as np


# # 2. 转换原始数据 

# In[2]:


#获取原始数据解压后的目录中的文件
#前提，把所有原始数据解压到一个文件夹中。
filename=os.listdir("/data/dataset/chinastock/data/")

#使用pandas处理数据
pdData=pd.DataFrame()
for eachfile in filename:
    filepath="/data/dataset/chinastock/data/"
    testfile=os.path.join(filepath,eachfile)
    try:
        #读取文件
        test=pd.read_csv(testfile,encoding="gb2312")
        # 生成心得列名称
        newcolumn=["Code","Name","Date","Industury","Concept","Region","Open","High",
            "Low","Close","PostRecovery","PreRecovery","Gain","Volumns","Trunover","HandTurnoverRate",
            "CirculationMarketValue","Capitalization","Toplimit","BottomLimit","PETTM","MSTTM","MRTTM",
            "PBRatio","MA_5","MA_10","MA_20","MA_30","MA_60","MA_GoldFork",
            "MACD_DIF","MACD_DEA","MACD_MACD","MACD_GoldFolk","KDJ_K",
            "KDJ_D","KDJ_J","KDJ_GoldFork","BollingerMid","BollingerUP","BollingerDown","psy",
            "psyma","rsi1","rsi2","rsi3","Amplitude","VolumnRatio"]
        test.columns=newcolumn
        slicedata=test
        # 转换时间列格式
        slicedata.loc[:,"Date"]=pd.to_datetime(slicedata.loc[:,"Date"])
        # 时间列作为索引
        slicedata=slicedata.set_index("Date",drop=True)
        slicedata=slicedata.sort_index(ascending=False)
        #微数据增加临时标签，先考虑二分类问题，按照涨跌分类。
        slicedata["label"]=slicedata.Gain>0
        slicedata.label.replace(True,1,inplace=True)
        #以0填充NA
        slicedata.fillna(0,inplace=True)
        #合并所有表格
        pdData=pd.concat([pdData,slicedata],axis=0)
    except:
        #忽略错误
        print(testfile)
        continue


# In[3]:


#保存结果到csv文件
pdData.to_csv("OriginalDataConcat.csv")
#数据转换为numpy形式，方便后续处理
data=np.array(pdData)


# In[4]:


#查看数据内容
pdData.head()


# # 3. 将数据标签分为8类

# In[6]:


## 将数据标签分为8类。
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
    if x>0.02 and x<=0.05:
        return 5
    if x>0.05 and x<=0.07:
        return 6
    if x>0.07:
        return 7


# # 4. 进一步简化数据

# In[7]:


from keras.utils import to_categorical
newpdData1=pd.read_csv("./OriginalDataConcat.csv")
targetcolumns=["Code","Date","Open","High","Low","Close","Gain","Volumns","MA_5","MA_10","MA_20"]
newlist=newpdData1.loc[:,targetcolumns]
newlist.head()


# In[8]:


newpdData1=newlist
# 生成新的8分类标签。
newpdData1["label"]=newpdData1["Gain"].apply(setlabel)
# 填充na
newpdData1.fillna(0,inplace=True)
#时间列格式转换
newpdData1.loc[:,"Date"]=pd.to_datetime(newpdData1.loc[:,"Date"])
newpdData1.head()


# In[9]:


#按照时间排序
newpdData1.sort_values(by=["Code","Date"],ascending=True,inplace=True)
#设置时间为索引
newpdData1=newpdData1.set_index("Date",drop=True)
#把股票代码作为int类型的参数
def changecodetoint(x):
    return int(x[2:])
newpdData1["Code"]=newpdData1["Code"].apply(changecodetoint)
newpdData1.head()


# In[10]:


#保存简化后的数据
newpdData1.to_csv("./SimpleVersionData.csv")
data=np.array(newpdData1)

