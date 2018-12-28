import numpy as np
import wfdb as wf
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import os, imageio
from tqdm import tqdm
import mitecg



ECG = mitecg.ReadMitEcg('/Users/chen/Desktop/ecg/www.physionet.org/physiobank/database/mitdb', 10000, [1, 2, 3, 4, 5], False)

notadd = [0, 0, 0, 0, 0]
typeName = ["Normal beat", "Left bundle branch block beat", "Right bundle branch block beat", "Aberrated atrial premature beat", "Premature ventricular contraction"]
for index in range(5):
    plt.figure()
    for i in range(5):
    
        batch_xs, batch_ys = ECG.nextbatch(20)
        for w in range(len(batch_ys)):
            if ((notadd[index]<5) and (batch_ys[w]==(index+1))):
                notadd[index] = notadd[index] + 1
                plt.subplot(510+notadd[index])
                plt.plot(batch_xs[w,:])
                plt.title(typeName[index])
    plt.savefig("realheatbeatsample/" + typeName[index] + ".png")



