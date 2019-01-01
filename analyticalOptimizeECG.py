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

SCALEDSAMPLES = 250
typeName = ["Normal beat", "Left bundle branch block beat", "Right bundle branch block beat", "Aberrated atrial premature beat", "Premature ventricular contraction"]


ECG = mitecg.ReadMitEcg('/Users/chen/Desktop/ecg/www.physionet.org/physiobank/database/mitdb', 10000, [1, 2, 3, 4, 5], False, SCALEDSAMPLES = SCALEDSAMPLES)

ecgarray_x, ecgarray_y = ECG.nextbatch(1)

print(ecgarray_x)
print(ecgarray_y)
for beat in range(len(ecgarray_y)):
    sampleArray = ecgarray_x[beat]
    print(sampleArray)
    print(typeName[ecgarray_y[beat]-1])
    x=np.linspace(0,1,sampleArray.size)

    yy=fft(sampleArray)
    print(yy)
    yreal = yy.real
    yimag = yy.imag

    yf=abs(yy)
    print(yf)
    yf1=abs(yy)/len(x)
    yf2 = yf1[range(int(len(x)/2))]




    for nu in range(len(yf)):
        if (yf[nu] < 3):
            yy[nu] = 0

    ifftsampleArray = ifft(yy)

    xf = np.arange(sampleArray.size)
    xf1 = xf
    xf2 = xf[range(int(len(x)/2))]



    plt.subplot(231)
    plt.plot(x,sampleArray)
    plt.title('Original wave ' + typeName[ecgarray_y[beat]-1])

    plt.subplot(232)
    plt.plot(xf,yf,'r')
    plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')
    plt.subplot(233)
    plt.plot(xf1,yf1,'g')
    plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

    plt.subplot(234)
    plt.plot(xf2,yf2,'b')
    plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')

    plt.subplot(235)
    plt.plot(xf,ifftsampleArray,'y')
    plt.title('iFFT wave)',fontsize=10,color='#F03030')


    plt.show()
#fft code reference to https://blog.csdn.net/ouening/article/details/71079535

#notadd = [0, 0, 0, 0, 0]
#typeName = ["Normal beat", "Left bundle branch block beat", "Right bundle branch block beat", "Aberrated atrial premature beat", "Premature ventricular contraction"]
#for index in range(5):
#    plt.figure()
#    for i in range(5):
#
#        batch_xs, batch_ys = ECG.nextbatch(20)
#        for w in range(len(batch_ys)):
#            if ((notadd[index]<5) and (batch_ys[w]==(index+1))):
#                notadd[index] = notadd[index] + 1
#                plt.subplot(510+notadd[index])
#                plt.plot(batch_xs[w,:])
#                plt.title(typeName[index])
#    plt.savefig("realheatbeatsample/" + typeName[index] + ".png")



