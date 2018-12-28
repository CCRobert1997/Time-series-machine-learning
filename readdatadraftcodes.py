import numpy as np
import wfdb as wf
import sys
import matplotlib.pyplot as plt

data = np.fromfile("/Users/chen/Desktop/ecg/www.physionet.org/physiobank/database/mitdb/105.dat", dtype=np.uint8)
head = open("/Users/chen/Desktop/ecg/www.physionet.org/physiobank/database/mitdb/105.hea", "r")
attr = np.fromfile("/Users/chen/Desktop/ecg/www.physionet.org/physiobank/database/mitdb/105.atr", dtype=np.uint8)


#head file
headread = head.readlines()
headfilelist = []
for line in headread:
   headfilelist.append(line.split())
#print(headfilelist)
nosig = int(headfilelist[0][1]) #number of channels
sfreq = int(headfilelist[0][2]) #frequency of sampling

SAMPLES2READ = 10000#2691


dformat = [212, 212] #initialize the format of the data, 212 is the only allowed here
gain = [200, 200] #the number of integers each mV have
bitres = [11, 11] #the precision of sampling
zerovalue = [1024, 1024] # the integer value of the zero point of the ECG signal
firstvalue =  [0, 0] #the first integer value of the signal
for i in range(int(nosig)):
    dformat[i] = int(headfilelist[i+1][1])
    gain[i] = int(headfilelist[i+1][2])
    bitres[i] = int(headfilelist[i+1][3])
    zerovalue[i] = int(headfilelist[i+1][4])
    firstvalue[i] = int(headfilelist[i+1][5])
#print([dformat, gain, bitres, zerovalue, firstvalue])

#data file
if (dformat != [212, 212]):
    sys.exit()
npdata = np.array(data[:(SAMPLES2READ*3)])
print(SAMPLES2READ)
npdata = np.resize(npdata, (SAMPLES2READ,3))
print(npdata[0,1])
print(npdata[1,2])
M2H = np.right_shift(npdata[:, 1], np.full(npdata[:, 1].size, 4))
M1H = np.bitwise_and(npdata[:, 1], np.full(npdata[:, 1].size, 15))

PRL = np.left_shift(np.bitwise_and(npdata[:, 1], np.full(npdata[:, 1].size, 8)), np.full(npdata[:, 1].size, 9))
PRR = np.left_shift(np.bitwise_and(npdata[:, 1], np.full(npdata[:, 1].size, 128)), np.full(npdata[:, 1].size, 5))
M = np.array([np.left_shift(M1H, np.full(M1H.size, 8)) + npdata[:, 0] - PRL, np.left_shift(M2H, np.full(M2H.size, 8)) + npdata[:, 2] - PRR])
M = np.transpose(M)
print((M[:, 0] - np.full(npdata[:, 0].size, zerovalue[0])) / gain[0])
print((M[:, 1] - np.full(npdata[:, 1].size, zerovalue[1])) / gain[1])
if ((M[0,:] != firstvalue).any()):
    print("inconsistency in the first bit values")
    sys.exit()
if (nosig==2):
    print(M)
    M0 = (M[:, 0] - np.full(npdata[:, 0].size, zerovalue[0])) / gain[0]
    M1 = (M[:, 1] - np.full(npdata[:, 1].size, zerovalue[1])) / gain[1]
    M = np.array([M0, M1], dtype = float)
    M = np.transpose(M)
    TIME = np.array(list(range(SAMPLES2READ)))/sfreq

    #print(M0.reshape(npdata[:, 0].size,1))
    #print(M1.reshape(npdata[:, 0].size,1))
    #np.insert(M, [0], M0.reshape(npdata[:, 0].size,1), axis=1)
    #np.insert(M, [1], M1.reshape(npdata[:, 0].size,1), axis=1)
elif (nosig==1):
    M0 = M[:, 0] - np.full(npdata[:, 0].size, zerovalue[0])
    M1 = M[:, 1] - np.full(npdata[:, 1].size, zerovalue[1])
    M = np.array([M0, M1], dtype = float)
    M = M.ravel(order='F')
    sM = M.size
    M[sM-1] = 0
    M = np.transpose(M)
    M = M/gain[0]
    TIME = np.array(list(range(2*SAMPLES2READ)))/sfreq
    print ("also true")
else:
    print ("false")
#print(M)
#print(TIME)
print("LOADING DATA FINISHED")





#attr file
ATRTIME=[]
ANNOT=[]
sa = attr.size
print(int(sa/2))
attr = np.reshape(attr, (-1, 2))
sa = attr.shape
print(sa)
saa = sa[0]
print(saa)
i = 0
while (i < saa):
    annoth = np.right_shift(attr[i, 1], 2)
    if (annoth==59):
        ANNOT.append(np.right_shift(attr[i+3, 1], 2))
        ATRTIME.append(attr[i+2, 0] + np.left_shift(attr[i+2, 1], 8) + np.left_shift(attr[i+1, 0], 16) + np.left_shift(attr[i+1, 1], 24))
        i = i + 3
        print(ANNOT)
    elif (annoth==63):
        hilfe = np.left_shift(np.bitwise_and(attr[i, 1], 3), 8) + attr[i, 0]
        hilfe = hilfe + hilfe%2
        i = i + int(hilfe/2)
    else:
        ATRTIME.append(np.left_shift(np.bitwise_and(attr[i, 1], 3), 8)+attr[i, 0])
        ANNOT.append(np.right_shift(attr[i, 1], 2))
    #elif (annoth==60):
    #elif (annoth==61):
    #elif (annoth==62):
    #else:
    i = i + 1
ATRTIME = np.cumsum(ATRTIME)/sfreq
ind = np.where(ATRTIME <= TIME[-1])
ATRTIMED = ATRTIME[ind]
ATRTIMED = np.insert(ATRTIMED, 0, 0)
#ANNOT=round(ANNOT)
ANNOTD= np.array(ANNOT)[ind]
print(ATRTIMED)
print(ind)
print(ANNOTD)


SCALEDSAMPLES = 250
j = 0
print(ATRTIMED[0])
print(ATRTIMED[1])
print(TIME)
while j < (len(ATRTIMED) - 1):
    if (ANNOTD[j] in [1, 2, 3, 4, 5]):
        print("vvvv")
    else:
        print("xxxx")
    indoneperiod = np.where((TIME >= ATRTIMED[j]) & (TIME < ATRTIMED[j+1]))
    heartBeatVoltage = M[:, 0][indoneperiod]
    heartBeatTime = TIME[indoneperiod]
    print(heartBeatTime)
    #print(M[:, 0][indoneperiod])
    #print(len(TIME[indoneperiod]))
    #print(ANNOTD[j])
    lines = plt.plot(heartBeatTime, heartBeatVoltage)#, TIME[indoneperiod], M[:, 1][indoneperiod])
    plt.xlabel("TIME")
    plt.title("ECG wave two chanels")
    plt.show()
    
    
    scaledSampleInterval = (ATRTIMED[j+1] - ATRTIMED[j])/SCALEDSAMPLES
    #print(scaledSampleInterval)
    k = indoneperiod[0][0] + 1
    #print(k)
    sampleID = 0
    scaledSampleList = []
    scaledTimeList = []
    #print(heartBeatTime[0])
    while (sampleID < SCALEDSAMPLES):
        while (((scaledSampleInterval*sampleID + TIME[indoneperiod[0][0]]) <= TIME[k]) and (sampleID < SCALEDSAMPLES)):
            
            lowT = TIME[k - 1]
            highT = TIME[k]
            scaledSampleList.append(((scaledSampleInterval*sampleID + TIME[indoneperiod[0][0]]) - lowT)/(highT-lowT) * (M[:, 0][k] - M[:, 0][k-1]) + M[:, 0][k-1])
            scaledTimeList.append(scaledSampleInterval*sampleID + TIME[indoneperiod[0][0]])
            #print(scaledTimeList[-1])
            sampleID = sampleID + 1
        #print("********")
        #print(TIME[k])
        #print("********************************8")
        k = k + 1


    #print(heartBeatTime[-1])
    #print(ATRTIMED[j+1])
    print(len(heartBeatTime))


    print(sampleID)
    lines = plt.plot(scaledTimeList, scaledSampleList)#, TIME[indoneperiod], M[:, 1][indoneperiod])
    plt.xlabel("TIME")
    plt.title("ECG wave two chanels")
    plt.show()
    j = j + 1



lines = plt.plot(TIME, M[:, 0], TIME, M[:, 1])
plt.xlabel("TIME")
plt.title("ECG wave two chanels")
plt.show()
lines = plt.plot(TIME, M[:, 0])
plt.xlabel("TIME")
plt.title("ECG wave single chanel")
plt.show()

