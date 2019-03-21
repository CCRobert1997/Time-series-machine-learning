import numpy as np
import wfdb as wf
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import os, imageio
from tqdm import tqdm
import mitecg
from scipy.fftpack import fft,ifft
import seaborn
from sklearn import metrics

SCALEDSAMPLES = 100
unnormalindex = 3

ECG = mitecg.ReadMitEcg('/Users/chen/Desktop/Research/ecg/www.physionet.org/physiobank/database/mitdb', 100000, [1,unnormalindex], False, SCALEDSAMPLES = SCALEDSAMPLES)

sampleArray = ECG.nextbatch_limitperclasssize(47, 5)

#print(sampleArray[0])
#comp1 = 0
#comp2 = 1
#print(sampleArray[1])
#print(sampleArray[1].size)
#for i in range(sampleArray[1].size):
#    if (sampleArray[1][i] == 1):
#        comp1 = comp1 + 1
#    if (sampleArray[1][i] == 3):
#        comp2 = comp2 + 1
#print(comp1)
#print(comp2)
negativeclass = sampleArray[1][0]
#for i in range(sampleArray[1].size):
#    if (sampleArray[1][i] == negativeclass):
#        sampleArray[1][i] = 0
#    else:
#        sampleArray[1][i] = 1

conca_data = np.c_[sampleArray[0], sampleArray[1]]
#print(conca_data)

#sigma = np.cov(conca_data.T)
sigma = np.corrcoef(conca_data.T)
cov11 = np.cov(sampleArray[0].T)
Dcov11 = np.diag(np.sqrt(cov11.diagonal()))

#print(sigma)
#print(np.cov(sampleArray[0].T))
#print(np.cov(sampleArray[0].T, sampleArray[1].T))
sigma11 = sigma[0:-1, 0:-1]
#print(sigma11)
sigma12 = sigma[0:-1,-1]
#print(sigma12)
sigma21 = sigma[-1,0:-1]
#print(sigma21)
sigma22 = sigma[-1,-1]
#print(sigma22)
#print(len(sigma21))



n = 1
xForSin = np.arange(0, SCALEDSAMPLES)/SCALEDSAMPLES
yForSin = np.sin(np.pi*2.*n*xForSin)
#plt.plot(xForSin, np.sin(np.pi*2.*n*xForSin))
#plt.show()
#print(sigma11[:,0])
#print(yForSin)
#print(sigma11[:,0] @ yForSin)
scale = np.sqrt(yForSin @ sigma11 @ yForSin)
print(scale)


n = 3
xForSin = np.arange(0, SCALEDSAMPLES)/SCALEDSAMPLES
yForSin = np.sin(np.pi*2.*n*xForSin)
scale = np.sqrt(yForSin @ sigma11 @ yForSin)
#print(scale)

#alpha = yForSin/scale
#print(np.sqrt(alpha @ sigma11 @ alpha))
#print(alpha)
gamma = 1/np.sqrt(sigma22)
#print(sigma22)
#print(gamma)
#print(alpha.transpose() @ sigma12 * gamma)

#test = yForSin @ sigma11
#print(test)
#print(test @ yForSin)
#print(sigma11)
#print(sigma11.shape)
shape_of_sigma11 = sigma11.shape
for i in range(shape_of_sigma11[0]):
    for j in range(shape_of_sigma11[1]):
        if not (sigma11[i][j] == sigma11[j][i]):
            sigma11[i][j] = sigma11[j][i]

covY = []
X = []
maxY = 0
wave_number_pick = 0
for m in range(150):
    X.append(m)
    xForSin = np.arange(0, SCALEDSAMPLES)/SCALEDSAMPLES
    yForSin = np.sin(np.pi*2.*m*xForSin)
    scale = np.sqrt(yForSin @ sigma11 @ yForSin)
    
    alpha = yForSin/scale

    gamma = 1/np.sqrt(sigma22)
    
    cov_for_n = alpha.transpose() @ sigma12 * gamma
    covY.append(cov_for_n)
    if (cov_for_n > maxY):
        cov_for_n = maxY
        wave_number_pick = m
        #yForSin
plt.plot(X, covY, 'ro')
plt.show()
print(wave_number_pick)


#test
testArray = ECG.nextbatch_limitperclasssize(47, 50)





forplot = 1
for i in range(testArray[1].size):
    if (testArray[1][i] == unnormalindex):
        if (forplot <= 9):
            plt.subplot(910+forplot)
            plt.plot(testArray[0][i,:])
            forplot = forplot + 1
plt.show()


forplot = 1
for i in range(testArray[1].size):
    
    if (forplot <= 9):
        plt.subplot(910+forplot)
        plt.plot(testArray[0][i,:])
        forplot = forplot + 1
plt.show()


for i in range(testArray[1].size):
    if (testArray[1][i] == negativeclass):
        testArray[1][i] = 1
    else:
        testArray[1][i] = 0




#wave_number_pick = 6
yForSin = np.sin(np.pi*2.*wave_number_pick*np.arange(0, SCALEDSAMPLES)/SCALEDSAMPLES)
pred = testArray[0] @ Dcov11 @ yForSin
print(1/(1 + np.exp(-pred)))
print(testArray[1])
fpr, tpr, thresholds = metrics.roc_curve(testArray[1], pred)
print(metrics.auc(fpr, tpr))
plt.plot(fpr, tpr, label='GBT')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()



