import pandas as pd
import numpy as np
import glob
from pathlib import Path
import re

df1 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie1996_Data_2023.csv")
df2 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie1998_Data_2023.csv")
df3 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2000_Data_2023.csv")
df4 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2002_Data_2023.csv")

print(list(df1))
print(list(df2))
print(list(df3))
print(list(df4))

df = df1.append(df2, ignore_index=True)
df = df.append(df3, ignore_index=True)
df = df.append(df4, ignore_index=True)
df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie9602_Data_2023paper.csv")

# df1 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2009_Data_2023paper.csv")
# df2 = pd.read_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie2010_Data_2023paper.csv")
#
#
# print(list(df1))
# print(list(df2))
#
#
# df = df1.append(df2, ignore_index=True)
#
# df.to_csv("/home/poyraden/Analysis/JOSIEfiles/Proccessed/Josie0910_Data_2023paper.csv")