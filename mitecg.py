import numpy as np
import wfdb as wf
import sys
import matplotlib.pyplot as plt
import os




class ReadMitEcg:
    def __init__(self, path, SAMPLES2READ_each_record=10000, heartBeatTypeLearn=[1, 2, 3, 4, 5], one_hot=True, SCALEDSAMPLES = 250):
        self.path = path
        self.SAMPLES2READ = SAMPLES2READ_each_record
        folder = os.fsencode(self.path)
        filenames = []
        #Delete 102-0.atr after download for easier data processing
        for file in os.listdir(folder):
            filename = os.fsdecode(file)
            if filename.endswith( ('.atr', '.dat', '.hea') ): # whatever file types you're using...
                filenames.append(path + '/' +filename)
        filenames.sort()
        for mm in filenames:
            print (mm)
        self.filenames = filenames
        self.heartBeatTypeLearn = heartBeatTypeLearn
        self.currentReadRecord = 0
        self.one_hot = one_hot
        self.SCALEDSAMPLES = SCALEDSAMPLES
    #batchsize is number of records but not number of heartbeats
    def nextbatch(self, batchsize):
        batch_xs = []
        batch_ys = []
        for b_i in range(batchsize):
            data = np.fromfile(self.filenames[(self.currentReadRecord)*3 + 1], dtype=np.uint8)
            head = open(self.filenames[(self.currentReadRecord)*3 + 2], "r")
            attr = np.fromfile(self.filenames[(self.currentReadRecord)*3 + 0], dtype=np.uint8)
            #print(self.filenames[(self.currentReadRecord + b_i)*3 + 1])
            #print(self.filenames[(self.currentReadRecord + b_i)*3 + 2])
            #print(self.filenames[(self.currentReadRecord + b_i)*3 + 0])
            
            self.currentReadRecord = self.currentReadRecord + 1
            if (self.currentReadRecord >= (len(self.filenames)/3 - 1)):
                self.currentReadRecord = 0
            #head file
            headread = head.readlines()
            headfilelist = []
            for line in headread:
                headfilelist.append(line.split())
            nosig = int(headfilelist[0][1]) #number of channels
            sfreq = int(headfilelist[0][2]) #frequency of sampling

            SAMPLES2READ = self.SAMPLES2READ

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
            head.close()
            #data file
            if (dformat != [212, 212]):
                sys.exit()
            npdata = np.array(data[:(SAMPLES2READ*3)])
            npdata = np.resize(npdata, (SAMPLES2READ,3))
            M2H = np.right_shift(npdata[:, 1], np.full(npdata[:, 1].size, 4))
            M1H = np.bitwise_and(npdata[:, 1], np.full(npdata[:, 1].size, 15))
            PRL = np.left_shift(np.bitwise_and(npdata[:, 1], np.full(npdata[:, 1].size, 8)), np.full(npdata[:, 1].size, 9))
            PRR = np.left_shift(np.bitwise_and(npdata[:, 1], np.full(npdata[:, 1].size, 128)), np.full(npdata[:, 1].size, 5))
            M = np.array([np.left_shift(M1H, np.full(M1H.size, 8)) + npdata[:, 0] - PRL, np.left_shift(M2H, np.full(M2H.size, 8)) + npdata[:, 2] - PRR])
            M = np.transpose(M)

            if ((M[0,:] != firstvalue).any()):
                print("inconsistency in the first bit values")
                sys.exit()
            if (nosig==2):
                M0 = (M[:, 0] - np.full(npdata[:, 0].size, zerovalue[0])) / gain[0]
                M1 = (M[:, 1] - np.full(npdata[:, 1].size, zerovalue[1])) / gain[1]
                M = np.array([M0, M1], dtype = float)
                M = np.transpose(M)
                TIME = np.array(list(range(SAMPLES2READ)))/sfreq
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
            else:
                print ("false")
            #print("LOADING the record " + self.filenames[(self.currentReadRecord- 1)*3 + 1]  + " FINISHED")

            # attr file
            ATRTIME = []
            ANNOT = []
            sa = attr.size
            attr = np.reshape(attr, (-1, 2))
            sa = attr.shape
            saa = sa[0]
            i = 0
            while (i < saa):
                annoth = np.right_shift(attr[i, 1], 2)
                if (annoth == 59):
                    ANNOT.append(np.right_shift(attr[i + 3, 1], 2))
                    ATRTIME.append(attr[i + 2, 0] + np.left_shift(attr[i + 2, 1], 8) + np.left_shift(attr[i + 1, 0],
                                                                                                     16) + np.left_shift(
                        attr[i + 1, 1], 24))
                    i = i + 3
                elif (annoth == 63):
                    hilfe = np.left_shift(np.bitwise_and(attr[i, 1], 3), 8) + attr[i, 0]
                    hilfe = hilfe + hilfe % 2
                    i = i + int(hilfe / 2)
                else:
                    ATRTIME.append(np.left_shift(np.bitwise_and(attr[i, 1], 3), 8) + attr[i, 0])
                    ANNOT.append(np.right_shift(attr[i, 1], 2))
                # elif (annoth==60):
                # elif (annoth==61):
                # elif (annoth==62):
                # else:
                i = i + 1
            ATRTIME = np.cumsum(ATRTIME) / sfreq
            ind = np.where(ATRTIME <= TIME[-1])
            ATRTIMED = ATRTIME[ind]
            ATRTIMED = np.insert(ATRTIMED, 0, 0)
            ANNOTD = np.array(ANNOT)[ind]
            #print(ANNOTD)
            #SCALEDSAMPLES = 250
            j = 0
            while j < (len(ATRTIMED) - 1):
                if (ANNOTD[j] in self.heartBeatTypeLearn):
                    indoneperiod = np.where((TIME >= ATRTIMED[j]) & (TIME < ATRTIMED[j + 1]))
                    heartBeatVoltage = M[:, 0][indoneperiod]
                    heartBeatTime = TIME[indoneperiod]
                    scaledSampleInterval = (ATRTIMED[j + 1] - ATRTIMED[j]) / self.SCALEDSAMPLES
                    # print(scaledSampleInterval)
                    k = indoneperiod[0][0] + 1
                    # print(k)
                    sampleID = 0
                    scaledSampleList = []
                    scaledTimeList = []
                    while (sampleID < self.SCALEDSAMPLES):
                        while (((scaledSampleInterval * sampleID + TIME[indoneperiod[0][0]]) <= TIME[k]) and (
                                sampleID < self.SCALEDSAMPLES)):
                            lowT = TIME[k - 1]
                            highT = TIME[k]
                            scaledSampleList.append(
                                ((scaledSampleInterval * sampleID + TIME[indoneperiod[0][0]]) - lowT) / (
                                            highT - lowT) * (M[:, 0][k] - M[:, 0][k - 1]) + M[:, 0][k - 1])
                            scaledTimeList.append(scaledSampleInterval * sampleID + TIME[indoneperiod[0][0]])
                            sampleID = sampleID + 1
                        k = k + 1

                    batch_xs.append(scaledSampleList)
                    heartBeatLabel = ANNOTD[j]
                    if (self.one_hot):
                        heartBeatLabel = [0] * len(self.heartBeatTypeLearn)
                        heartBeatLabel[self.heartBeatTypeLearn.index(ANNOTD[j])] = 1
                    batch_ys.append(heartBeatLabel)
                j = j + 1
        return np.array(batch_xs), np.array(batch_ys)


    def oneEcgWithHeartBeatScaled(self):
        ecgarray_x, ecgarray_y = self.nextbatch(1)
        resultarray = ecgarray_x.reshape((1, ecgarray_x.size))[0]
        return resultarray



    








