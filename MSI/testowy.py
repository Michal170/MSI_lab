# Zaimportuj bibliotekę numpy
import numpy as np
import pandas as pd
import sklearn as sk

#pandas
a = pd.Series([-1,1,3,5,7])
aa = a*10
print(aa)

#sklearn

#1
table = np.ones([3,5])
print(table)
#2
rsum = np.sum(table,axis=0)
csum = np.sum(table,axis=1)
print(rsum)
print(csum)
#3
table_3 = np.arange(20)
print(table_3)
#4
print(table_3[10:13])












# a = np.array([1,3,5,7])
#
# b = np.arange(4)
#
# np.arange(2,10,1)
#
# np.linspace(0,10,6)
# c= np.array([[1,2,3],[4,5,6]])
#
# d = np.ones((3,5))
#
# ab= a+b
# print("macierz c: \n",c)
# print("macierz d: \n",d)
# print("suma po kolumnach:\n",ab)
#
# np.transpose(c)
# np.transpose(d)
# print("Tmacierz c:\n",c)
# print("Tmacierz d:\n",d)
# ab=a+b
# print("macierz zsumowana: ab\n",ab)