import struct
import numpy as np
import copy
from . import Overfitting
import random as rand

WIDTH = 146
HEIGHT = 122

KToCelsius = -273.15
lowerTemp = 150


class CNNDataset:
    def __init__(self, timelist=[6, 8, 12], obslist=["PM25"], obsTimeList=[i for i in range(1, 6, 1)], obsMax=[190],
                 obsMin=[0], forelist=["PM25"], foreTimeList=[6, 8, 12], foreMax=[200], foreMin=[0], obs=False,
                 fore=False, CNNPath="./data/CNN", DNNPath="./data/", train=True, inputNum=42, databalancing=False,
                 numBalanceData=0, dataType="PM2.5",lstmHiddenDim = 15):
        self.timelist = timelist
        self.currentLearnTime = 0
        self.currentBatch = 0
        self.lstmHiddenDim = lstmHiddenDim
        self.obsFlag = obs
        self.obslist = obslist
        self.obsTimeList = obsTimeList
        self.obsMax = obsMax
        self.obsMin = obsMin

        self.foreFlag = fore
        self.forelist = forelist
        self.foreTimeList = foreTimeList
        self.foreMax = foreMax
        self.foreMin = foreMin

        self.train = train
        self.CNNPath = CNNPath
        self.DNNPath = DNNPath
        self.inputNum = inputNum

        if self.train == True:
            self.CNNPath = self.CNNPath + "Learn/"
            for i in range(0, len(self.DNNPath)):
                self.DNNPath[i] = self.DNNPath[i]
        else:
            self.CNNPath = self.CNNPath + "Predict/"
            #self.DNNPath = self.DNNPath + "Predict/"
            for i in range(0, len(self.DNNPath)):
                self.DNNPath[i] = self.DNNPath[i]+"Predict/"

        self.SynchronizationDate()

        self.obsData = np.zeros((len(self.obslist), len(obsTimeList), len(self.combineDate), HEIGHT * WIDTH),                # 2021.06.28 kT inf.// obsTimeList: T1~5, combineDate: 최종적인 학습에 사용되는 날짜
                                dtype=np.float32)
        self.foreData = np.zeros((len(self.forelist), len(foreTimeList), len(self.combineDate), HEIGHT * WIDTH),             # 2021.06.28 kT inf.// foreTimeList: T6~15
                                 dtype=np.float32)

        self.ReadDataDNN()
        self.ReadDataCNN()
        self.dataBalancing = databalancing
        if databalancing == True:
            if numBalanceData == 0:
                print("Warning : Current numBalanceData is Zero, Not Create Data")
            self.dataType = dataType
            self.numBalanceData = numBalanceData
            self.CallDataBalancing()

    def __len__(self):
        return self.len

    def InitBatch(self):
        self.currentBatch = 0

    def EndLearn(self):
        self.currentLearnTime += 1
        if self.dataBalancing == True and self.currentLearnTime < len(self.timelist):
            self.CallDataBalancing()

    def CallDataBalancing(self):
        self.BalancingDate, self.BalancingTable = Overfitting.DataBalance(self.DNN_y[0][self.currentLearnTime],
                                                                          self.combineDate, PMType=self.dataType)
        self.BalancingDate, self.BalancingTable = self.dataShuffle(self.BalancingDate, self.BalancingTable)
        self.len = len(self.BalancingTable)

    def ReadDataCNN(self):
        if self.foreFlag == True:
            for i in range(len(self.forelist)):
                for j in range(len(self.foreTimeList)):
                    self.ReadDataFore(i, j)

        if self.obsFlag == True:
            for i in range(len(self.obslist)):
                for j in range(len(self.obsTimeList)):
                    self.ReadDataObs(i, self.obsTimeList[j])

    def ReadDataDNN(self):

        self.DNN_x = np.zeros((len(self.DNNPath), len(self.timelist), self.len, self.inputNum), dtype=np.float32)             # 2021.06.28 kT inf.// self.timelist : T6~15
        #self.DNN_y = np.zeros((len(self.DNNPath), len(self.timelist), self.len, 1), dtype=np.float32) - 1#2021.06.29 MU // 원본, T6~T15까지 개별 입력
        self.DNN_y = np.zeros((len(self.DNNPath), len(self.timelist), self.len, 10), dtype=np.float32) - 1#2021.06.29 MU // 원본, T6~T15까지 통합 입력
        #self.DNN_y = np.zeros((len(self.DNNPath), len(self.timelist), self.len, 1),dtype=np.float32) - 1  # 2021.07.08 MU // 원본, T6~T15까지 통합 입력, T6만 출력

        if self.train == True:
            dataName = "Learn"
        else:
            dataName = "Predict"
        for r in range(len(self.DNNPath)):
            #for i in range(len(self.timelist)):
            for i in range(len([6])):

                temp = np.loadtxt(self.DNNPath[r] + "T06_T15_Max_Learn.txt",dtype=np.float32,delimiter=" ")#2021.07.07 MU // 학습시
                #temp = np.loadtxt(self.DNNPath[r] + "T06_T15_Max_Predict.txt",dtype=np.float32,delimiter=" ")  # 2021.07.07 MU // 테스트시

                #temp = np.loadtxt(self.DNNPath[r] + "T%02d_Max_" % self.timelist[i] + dataName + ".txt", # 2021.06.29 KT // 새로운 입력데이터를 위해 기존 데이터 삭제
                #                  dtype=np.float32,# 2021.06.29 KT // 새로운 입력데이터를 위해 기존 데이터 삭제
                #                  delimiter=" ")## 2021.06.29 KT // 새로운 입력데이터를 위해 기존 데이터 삭제
                index = 0
                for index in range(len(self.combineDate)):
                    for j in range(len(temp)):
                        if self.combineDate[index] == self.flatDate[r][j]:
                            self.DNN_x[r][i][index] = copy.deepcopy(temp[j][:-10])#2021.06.29 MU // T6~T15 데이터 병합위해 삭제, 데이터
                            #self.DNN_y[r][i][index] = copy.deepcopy(temp[j][-10])#2021.07.08 MU // T6 결과만 예측하기 위해 데이터 구성
                            self.DNN_y[r][i][index] = copy.deepcopy(temp[j][-10:])  # 2021.06.29 MU // T6~T15 데이터 병합위해 삭제, 데이터

                            #                            self.DNN_x[r][i][index] = copy.deepcopy(temp[j][:-1])#2021.06.29 MU // T6~T15 데이터 병합위해 삭제, 데이터
#                            self.DNN_y[r][i][index] = copy.deepcopy(temp[j][-1])#2021.06.29 MU // T6~T15 데이터 병합위해 삭제, 라벨

                            break

                        #index += 1

    # 2021.06.28 KT 삭제 // T6만 불러오는게 아니라 T6~15한번에 부르기 위해서 삭제후 다시 작성
    ''' 
    def ReadDataFore(self, foreIndex, foreTime):
        print(self.CNNPath + "T%02d_fore_" % self.foreTimeList[foreTime] + "%s" % self.forelist[
            foreIndex] + ".dat start Read")
        with open(self.CNNPath + "T%02d_fore_" % self.foreTimeList[foreTime] + "%s" % self.forelist[foreIndex] + ".dat",
                  "rb") as f:
            numpattern = len(self.foreDate)
            width = struct.unpack("i", f.read(4))[0]
            height = struct.unpack("i", f.read(4))[0]
            print(width, height)
            index = 0

            # print(len(self.foreData))
            for i in range(numpattern):
                if index >= self.len:
                    break
                if self.foreDate[i] != self.combineDate[index]:
                    f.seek(width * height * 4)
                    continue

                for j in range(height * width):
                    temp = struct.unpack("f", f.read(4))[0]

                    if self.forelist[foreIndex] == "TA" and temp > lowerTemp:
                        temp += KToCelsius

                    if temp > self.foreMax[foreIndex]:
                        temp = self.foreMax[foreIndex]
                    elif temp < self.foreMin[foreIndex]:
                        temp = self.foreMin[foreIndex]

                    if self.foreMin[foreIndex] < 0:
                        self.foreData[foreIndex][foreTime][index][j] = temp / (
                                self.foreMax[foreIndex] + abs(self.foreMin[foreIndex]))
                    else:
                        self.foreData[foreIndex][foreTime][index][j] = (temp - self.foreMin[foreIndex]) / (
                                self.foreMax[foreIndex] - self.foreMin[foreIndex])
                index += 1
        print(self.CNNPath + "T%02d_fore_" % self.foreTimeList[foreTime] + "%s" % self.forelist[foreIndex] + ".dat End")
    '''

    # 2021.06.28 KT 삭제 // T6만 불러오는게 아니라 T6~15한번에 부르기 위해서 삭제후 다시 작성
    def ReadDataFore(self, foreIndex, foreTime):
        print(self.CNNPath + "T%02d_fore_" % self.foreTimeList[foreTime] + "%s" % self.forelist[
            foreIndex] + ".dat start Read")
        with open(self.CNNPath + "T%02d_fore_" % self.foreTimeList[foreTime] + "%s" % self.forelist[foreIndex] + ".dat",
                  "rb") as f:
            numpattern = len(self.foreDate)
            width = struct.unpack("i", f.read(4))[0]
            height = struct.unpack("i", f.read(4))[0]
            print(width, height)
            index = 0

            # print(len(self.foreData))
            for i in range(numpattern):
                if index >= self.len:
                    break
                if self.foreDate[i] != self.combineDate[index]:
                    f.seek(width * height * 4)
                    continue

                for j in range(height * width):
                    temp = struct.unpack("f", f.read(4))[0]

                    if self.forelist[foreIndex] == "TA" and temp > lowerTemp:
                        temp += KToCelsius

                    if temp > self.foreMax[foreIndex]:
                        temp = self.foreMax[foreIndex]
                    elif temp < self.foreMin[foreIndex]:
                        temp = self.foreMin[foreIndex]

                    if self.foreMin[foreIndex] < 0:
                        self.foreData[foreIndex][foreTime][index][j] = temp / (
                                self.foreMax[foreIndex] + abs(self.foreMin[foreIndex]))
                    else:
                        self.foreData[foreIndex][foreTime][index][j] = (temp - self.foreMin[foreIndex]) / (
                                self.foreMax[foreIndex] - self.foreMin[foreIndex])
                index += 1
        print(self.CNNPath + "T%02d_fore_" % self.foreTimeList[foreTime] + "%s" % self.forelist[foreIndex] + ".dat End")



    def ReadDataObs(self, obsIndex, obsTime):
        print(self.CNNPath + "T%02d_obs_" % self.obsTimeList[obsTime - 1] + "%s" % self.obslist[
            obsIndex] + ".ubyte start Read")
        with open(
                self.CNNPath + "T%02d_obs_" % self.obsTimeList[obsTime - 1] + "%s" % self.obslist[obsIndex] + ".ubyte",
                "rb") as f:

            numpattern = struct.unpack("i", f.read(4))[0] # 2021.06.28 KT //  코드 수정[  len(self.obsDate)  ]하려고 했으나 index error  문제가 뜸 원래대로 교체
            width = struct.unpack("i", f.read(4))[0]
            height = struct.unpack("i", f.read(4))[0]
            print(numpattern, width, height)
            index = 0

            print(len(self.obsData))
            for i in range(numpattern):
                if index >= self.len:
                    break
                if self.obsDate[i] != self.combineDate[index]:
                    f.seek(width * height * 4)
                    continue

                for j in range(height * width):
                    temp = struct.unpack("f", f.read(4))[0]

                    if self.obslist[obsIndex] == "TA" and temp > lowerTemp:
                        temp += KToCelsius

                    if temp > self.obsMax[obsIndex]:
                        temp = self.obsMax[obsIndex]
                    elif temp < self.obsMin[obsIndex]:
                        temp = self.obsMin[obsIndex]

                    if self.obsMin[obsIndex] < 0:
                        self.obsData[obsIndex][obsTime - 1][index][j] = temp / (
                                self.obsMax[obsIndex] + abs(self.obsMin[obsIndex]))
                    else:
                        self.obsData[obsIndex][obsTime - 1][index][j] = (temp - self.obsMin[obsIndex]) / (
                                self.obsMax[obsIndex] - self.obsMin[obsIndex])

                index += 1
        print(
            self.CNNPath + "T%02d_obs_" % self.obsTimeList[obsTime - 1] + "%s" % self.obslist[obsIndex] + ".ubyte End")

    def SynchronizationDate(self):
        self.flatDate = [[] for _ in range(len(self.DNNPath))]  # DNN데이터 날짜

        self.obsDate = np.array([], dtype=np.int32)  # 측정데이터 날짜
        self.foreDate = np.array([], dtype=np.int32)  # 예보데이터 날짜
        self.combineDate = np.array([], dtype=np.int32)  # 공통 날짜
        self.len = 0

        print("Start Date Synchronization")
        #self.DNNPath = [self.DNNPath]   # train일 경우 주석, test일 경우 주석풀기
        for i in range(len(self.DNNPath)):
            with open(self.DNNPath[i] + "DateFlat.txt", "r") as f:
                for date in f:
                    self.flatDate[i].append(int(date[:]))
        # print(self.flatDate)
        if self.obsFlag == True:

            with open(self.CNNPath + "DateObs.txt", "r") as f:
                for date in f:
                    self.obsDate = np.append(self.obsDate, int(date[:]))
        # print(self.obsDate)
        if self.foreFlag == True:

            with open(self.CNNPath + "DateFore.txt", "r") as f:
                for date in f:
                    self.foreDate = np.append(self.foreDate, int(date[:]))

        # print(self.foreDate)

        print(len(self.flatDate), len(self.obsDate), len(self.foreDate))
        tempDate = np.array([])
        i = [0 for i in range(len(self.DNNPath))]
        j = 0
        # 1) Flat vs Obs
        while (1):
            if j >= len(self.obsDate):
                break
            if self.obsFlag == True:
                # if j >= len(self.obsDate):
                #     break
                for r in range(len(self.DNNPath)):
                    if self.flatDate[r][i[r]] == self.obsDate[j]:
                        # if not len(np.where(tempDate==int(self.obsDate[j]))):
                        if not np.any(tempDate == int(self.obsDate[j])):
                            tempDate = np.append(tempDate, int(self.obsDate[j]))
                        i[r] += 1

                    # elif self.flatDate[r][i[r]] > self.obsDate[j]:
                    #     #if not len(np.where(tempDate==int(self.obsDate[j]))):
                    #     if not np.any(tempDate == int(self.obsDate[j])):
                    #         tempDate = np.append(tempDate, int(self.obsDate[j]))

                    # elif self.flatDate[r][i[r]] < self.obsDate[j]:
                    # if not len(np.where(tempDate==int(self.obsDate[j]))):
                    if i[r] < len(self.flatDate[r]) - 1:
                        while (self.flatDate[r][i[r]] <= self.obsDate[j]):
                            if self.flatDate[r][i[r]] == self.obsDate[j]:
                                if not np.any(tempDate == int(self.obsDate[j])):
                                    tempDate = np.append(tempDate, int(self.obsDate[j]))
                                break
                            if i[r] >= len(self.flatDate[r]) - 1:
                                break
                            i[r] += 1
                j += 1
            elif self.obsFlag == False:
                tempDate = np.append(tempDate, int(self.flatDate[i]))
                i += 1
        print(len(tempDate))
        i = 0
        j = 0
        # 2) 1)'s result vs Fore
        while (1):
            if i >= len(tempDate):
                break
            if self.foreFlag == True:
                if j >= len(self.foreDate):
                    break
                if tempDate[i] == self.foreDate[j]:
                    self.combineDate = np.append(self.combineDate, int(tempDate[i]))
                    i += 1
                    j += 1

                elif tempDate[i] > self.foreDate[j]:
                    j += 1
                elif tempDate[i] < self.foreDate[j]:
                    i += 1
            elif self.foreFlag == False:
                self.combineDate = np.append(self.combineDate, int(tempDate[i]))
                i += 1
        self.len = len(self.combineDate)
        np.sort(self.combineDate)
        print(len(self.combineDate))
        # for i in range(len(self.combineDate)) :
        #    print(self.combineDate[i])

        print("End Date Synchronization")

    def NextBatchFlat(self, batchSize, random=False):

        if batchSize > (self.len - self.currentBatch):
            batchSize = self.len - self.currentBatch

        temp_x = np.zeros((batchSize, self.inputNum))
        temp_y = np.zeros((batchSize, 1))
        temp_index = np.zeros((batchSize))

        if random == True:
            randomNumber = np.random.randint(low=0, high=self.len, size=batchSize)
            for i in range(batchSize):
                temp_x[i] = copy.deepcopy(self.DNN_x[self.currentLearnTime][randomNumber[i]])
                temp_y[i] = copy.deepcopy(self.DNN_y[self.currentLearnTime][randomNumber[i]])
            temp_index = copy.deepcopy(randomNumber)
        else:
            index = 0
            for i in range(self.currentBatch, self.currentBatch + batchSize, 1):
                temp_x[index] = copy.deepcopy(self.DNN_x[self.currentLearnTime][i])
                temp_y[index] = copy.deepcopy(self.DNN_y[self.currentLearnTime][i])
                temp_index[index] = i
                index += 1

        self.currentBatch += batchSize

        return temp_x, temp_y, temp_index

    def NextBatchAll(self, batchSize, random=False, train=True):

        if self.dataBalancing == True and train == True:
            timeTable = copy.deepcopy(self.BalancingTable)
            dateTable = copy.deepcopy(self.BalancingDate)
        else:
            timeTable = np.arange(0, len(self.combineDate), 1, dtype=np.int32)

        if batchSize > (len(timeTable) - self.currentBatch):
            batchSize = len(timeTable) - self.currentBatch

        temp_x = np.zeros((batchSize, self.inputNum))
        #temp_y = np.zeros((batchSize, 1))#2021.06.29 MU // T6~T15 개별 학습시, 원본
        temp_y = np.zeros((batchSize, 10))#2021.06.29 MU // T6~T15 통합 학습시, 수정본
        temp_initenccell = np.zeros((batchSize, HEIGHT, WIDTH, self.lstmHiddenDim), dtype=np.float64)
        temp_initenchidden = np.zeros((batchSize, HEIGHT, WIDTH, self.lstmHiddenDim), dtype=np.float64)
        temp_initdeccell = np.zeros((batchSize, HEIGHT, WIDTH, self.lstmHiddenDim), dtype=np.float64)
        temp_initdechidden = np.zeros((batchSize, HEIGHT, WIDTH, self.lstmHiddenDim), dtype=np.float64)

        temp_fore = np.zeros((len(self.forelist), batchSize, WIDTH * HEIGHT)) # 2021.06.28 KT 아래 코드로 수정
        temp_fore = np.zeros((len(self.forelist), batchSize, len(self.foreTimeList), WIDTH * HEIGHT)) # 2021.06.28 KT 코드 수정

        temp_obs = np.zeros((len(self.obslist), batchSize, len(self.obsTimeList), WIDTH * HEIGHT))

        temp_index = np.zeros((batchSize))

        if random == True:
            randomNumber = np.random.randint(low=0, high=len(timeTable), size=batchSize)
            for i in range(batchSize):
                randomIndex = timeTable[randomNumber[i]]            # 2021.06.28 KT : 전체 label 갯수
                temp_index[i] = copy.deepcopy(randomIndex)
                random_region_list = []
                for r in range(len(self.DNNPath)):
                    if self.DNN_y[r][self.currentLearnTime][randomIndex].all() != -1:#
                        random_region_list.append(r)
                if not random_region_list:
                    print(randomIndex)
                random_region = rand.choice(random_region_list)
                #print("self.DNN_x[random_region][self.currentLearnTime]",self.DNN_x[random_region][self.currentLearnTime])
                temp_x[i] = copy.deepcopy(self.DNN_x[random_region][self.currentLearnTime][randomIndex])#나중에 수정필요
                temp_y[i] = copy.deepcopy(self.DNN_y[random_region][self.currentLearnTime][randomIndex])##나중에 수정필요

                # temp_x = np.append(temp_x,self.DNN_x[self.currentLearnTime][randomIndex])
                # temp_y = np.append(temp_y,self.DNN_y[self.currentLearnTime][randomIndex])
#                for j in range(len(self.forelist)):
#                    temp_fore[j][i] = copy.deepcopy(self.foreData[j][self.currentLearnTime][randomIndex])
                for j in range(len(self.forelist)):
                    for k in range(len(self.foreTimeList)):                         # 2021.06.28 KT //  추가
                        temp_fore[j][i][k] = copy.deepcopy(self.foreData[j][k][randomIndex]) # 2021.06.28 KT //  추가 self.currentLearnTime -> k
                for j in range(len(self.obslist)):
                    for k in range(len(self.obsTimeList)):              # 2021.06.28 KT // self.obsTimeList :T1~T5
                        temp_obs[j][i][k] = copy.deepcopy(self.obsData[j][k][randomIndex])

        else:
            index = 0
            for i in range(self.currentBatch, self.currentBatch + batchSize, 1):
                timeIndex = timeTable[i]
                temp_x[index] = copy.deepcopy(self.DNN_x[0][self.currentLearnTime][timeIndex])
                temp_y[index] = copy.deepcopy(self.DNN_y[0][self.currentLearnTime][timeIndex])
                temp_index[index] = timeIndex
                for j in range(len(self.forelist)):
                    temp_fore[j][index] = copy.deepcopy(self.foreData[j][self.currentLearnTime][timeIndex])
                for j in range(len(self.obslist)):
                    for k in range(len(self.obsTimeList)):
                        temp_obs[j][index][k] = copy.deepcopy(self.obsData[j][k][timeIndex])
                index += 1
        self.currentBatch += batchSize
        #print("temp_obs.shape", temp_obs.shape)
        #temp_obs.shape = (data종류, batchsize,timeindex, width*height)
        temp_obs = temp_obs.swapaxes(1,0)#(batchsize,data종류,timeindex, width*height)
        temp_obs = temp_obs.swapaxes(1,3)#(batchsize, width*height,timeindex,data종류)
        temp_obs = temp_obs.swapaxes(1,2)#(batchsize, timeindex, width*height,data종류)
        #print("temp_obs.shape", temp_obs.shape)
        #print("temp_obs.type", temp_obs.dtype)
        return temp_x, temp_y, temp_fore, temp_obs, temp_initenccell, temp_initenchidden, temp_initdeccell, temp_initdechidden, temp_index
        #return temp_x, temp_y, temp_fore, temp_obs, temp_index

    def ReturnAllData(self, random=False, train=True):
        if self.dataBalancing == True and train == True:
            timeTable = copy.deepcopy(self.BalancingTable)
            dateTable = copy.deepcopy(self.BalancingDate)
        else:
            timeTable = np.arange(0, len(self.combineDate), 1, dtype=np.int32)

        batchSize = len(self.DNNPath)*len(timeTable)

        temp_x = np.zeros((batchSize, self.inputNum))
        # temp_y = np.zeros((batchSize, 1))#2021.06.29 MU // T6~T15 개별 학습시, 원본
        temp_y = np.zeros((batchSize, 10))  # 2021.06.29 MU // T6~T15 통합 학습시, 수정본
        temp_initenccell = np.zeros((batchSize, HEIGHT, WIDTH, self.lstmHiddenDim), dtype=np.float64)
        temp_initenchidden = np.zeros((batchSize, HEIGHT, WIDTH, self.lstmHiddenDim), dtype=np.float64)
        temp_initdeccell = np.zeros((batchSize, HEIGHT, WIDTH, self.lstmHiddenDim), dtype=np.float64)
        temp_initdechidden = np.zeros((batchSize, HEIGHT, WIDTH, self.lstmHiddenDim), dtype=np.float64)

        #temp_fore = np.zeros((len(self.forelist), batchSize, WIDTH * HEIGHT))  # 2021.06.28 KT 아래 코드로 수정
        temp_fore = np.zeros(
            (len(self.forelist), batchSize, len(self.foreTimeList), WIDTH * HEIGHT))  # 2021.06.28 KT 코드 수정

        temp_obs = np.zeros((len(self.obslist), batchSize, len(self.obsTimeList), WIDTH * HEIGHT))

        temp_index = np.zeros((batchSize))

        index = 0

        for region in range(len(self.DNNPath)):
            for i in range(len(timeTable)):
                timeIndex = timeTable[i]
                temp_x[index] = copy.deepcopy(self.DNN_x[region][self.currentLearnTime][timeIndex])
                temp_y[index] = copy.deepcopy(self.DNN_y[region][self.currentLearnTime][timeIndex])
                temp_index[index] = timeIndex
                for j in range(len(self.forelist)):
                    temp_fore[j][index] = copy.deepcopy(self.foreData[j][self.currentLearnTime][timeIndex])
                for j in range(len(self.obslist)):
                    for k in range(len(self.obsTimeList)):
                        temp_obs[j][index][k] = copy.deepcopy(self.obsData[j][k][timeIndex])
                index += 1


        self.currentBatch += batchSize
        # print("temp_obs.shape", temp_obs.shape)
        # temp_obs.shape = (data종류, batchsize,timeindex, width*height)
        temp_obs = temp_obs.swapaxes(1, 0)  # (batchsize,data종류,timeindex, width*height)
        temp_obs = temp_obs.swapaxes(1, 3)  # (batchsize, width*height,timeindex,data종류)
        temp_obs = temp_obs.swapaxes(1, 2)  # (batchsize, timeindex, width*height,data종류)
        # print("temp_obs.shape", temp_obs.shape)
        # print("temp_obs.type", temp_obs.dtype)
        return temp_x, temp_y, temp_fore, temp_obs, temp_initenccell, temp_initenchidden, temp_initdeccell, temp_initdechidden, temp_index
        # return temp_x, temp_y, temp_fore, temp_obs, temp_index

    def NewNextBatchAll(self, obsData, foreData, flatData, targetData, batchSize, random=False):
        if batchSize > (self.len - self.currentBatch):
            batchSize = self.len - self.currentBatch

        temp_x = np.zeros((batchSize, self.inputNum))
        temp_y = np.zeros((batchSize, 1))

        temp_fore = np.zeros((len(self.forelist), batchSize, WIDTH * HEIGHT))

        temp_obs = np.zeros((len(self.obslist), batchSize, len(self.obsTimeList), WIDTH * HEIGHT))

        temp_index = np.zeros((batchSize))

        if random == True:
            randomNumber = np.random.randint(low=0, high=len(targetData), size=batchSize)
            for i in range(batchSize):
                temp_x[i] = copy.deepcopy(flatData[0][randomNumber[i]])
                temp_y[i] = copy.deepcopy(targetData[0][randomNumber[i]])
                for j in range(len(self.forelist)):
                    temp_fore[j][i] = copy.deepcopy(foreData[j][0][randomNumber[i]])
                for j in range(len(self.obslist)):
                    for k in range(len(self.obsTimeList)):
                        temp_obs[j][i][k] = copy.deepcopy(obsData[j][k][randomNumber[i]])
            temp_index = copy.deepcopy(randomNumber)
        else:
            index = 0
            for i in range(self.currentBatch, self.currentBatch + batchSize, 1):
                temp_x[index] = copy.deepcopy(flatData[0][i])
                temp_y[index] = copy.deepcopy(targetData[0][i])
                temp_index[index] = i
                for j in range(len(self.forelist)):
                    temp_fore[j][index] = copy.deepcopy(foreData[j][0][i])
                for j in range(len(self.obslist)):
                    for k in range(len(self.obsTimeList)):
                        temp_obs[j][index][k] = copy.deepcopy(obsData[j][k][i])
                index += 1
        self.currentBatch += batchSize

        return temp_x, temp_y, temp_fore, temp_obs, temp_index

    def NextBatchBalance(self, timeTable, batchSize, random=False):
        if batchSize > (len(timeTable) - self.currentBatch):
            batchSize = len(timeTable) - self.currentBatch

        temp_x = np.zeros((batchSize, self.inputNum))
        temp_y = np.zeros((batchSize, 1))

        temp_fore = np.zeros((len(self.forelist), batchSize, WIDTH * HEIGHT))
        temp_obs = np.zeros((len(self.obslist), batchSize, len(self.obsTimeList), WIDTH * HEIGHT))
        temp_index = np.zeros((batchSize))

        if random == True:
            randomNumber = np.random.randint(low=0, high=len(timeTable), size=batchSize)
            for i in range(batchSize):
                randomIndex = timeTable[randomNumber[i]]
                temp_index[i] = copy.deepcopy(randomIndex)
                temp_x[i] = copy.deepcopy(self.DNN_x[self.currentLearnTime][randomIndex])
                temp_y[i] = copy.deepcopy(self.DNN_y[self.currentLearnTime][randomIndex])
                for j in range(len(self.forelist)):
                    temp_fore[j][i] = copy.deepcopy(self.foreData[j][self.currentLearnTime][randomIndex])
                for j in range(len(self.obslist)):
                    for k in range(len(self.obsTimeList)):
                        temp_obs[j][i][k] = copy.deepcopy(self.obsData[j][k][randomIndex])
        else:
            index = 0

            for i in range(self.currentBatch, self.currentBatch + batchSize, 1):
                timeIndex = timeTable[i]
                temp_x[index] = copy.deepcopy(self.DNN_x[self.currentLearnTime][timeIndex])
                temp_y[index] = copy.deepcopy(self.DNN_y[self.currentLearnTime][timeIndex])
                temp_index[index] = timeIndex
                for j in range(len(self.forelist)):
                    temp_fore[j][index] = copy.deepcopy(self.foreData[j][self.currentLearnTime][timeIndex])
                for j in range(len(self.obslist)):
                    for k in range(len(self.obsTimeList)):
                        temp_obs[j][index][k] = copy.deepcopy(self.obsData[j][k][timeIndex])
                index += 1
        self.currentBatch += batchSize
        return temp_x, temp_y, temp_fore, temp_obs, temp_index

    def NextBatchObs(self):
        pass

    def dataShuffle(self, date, dateTable):

        for i in range(len(date * 2)):
            randomNumber = np.random.randint(low=0, high=len(date), size=2)

            temp = copy.deepcopy(date[randomNumber[0]])
            date[randomNumber[0]] = copy.deepcopy(date[randomNumber[1]])
            date[randomNumber[1]] = copy.deepcopy(temp)

            temp = copy.deepcopy(dateTable[randomNumber[0]])
            dateTable[randomNumber[0]] = copy.deepcopy(dateTable[randomNumber[1]])
            dateTable[randomNumber[1]] = copy.deepcopy(temp)

        return copy.deepcopy(date), copy.deepcopy(dateTable)

    def HighPMValueIncrease(self, PMType='PM2.5'):
        badValue = 0
        maxPM = 0

        if PMType == "PM2.5":
            badValue = 35.5
            maxPM = 110.0
        elif PMType == "PM10":
            badValue = 80.5
            maxPM = 180.0

        for i in range(len(self.timelist)):
            for j in range(len(self.combineDate)):
                temp = round(self.DNN_y[i][j][0] * maxPM)
                if temp >= badValue:
                    IncreaseData = (temp * 1.1)
                    if IncreaseData > maxPM:
                        IncreaseData = maxPM
                    self.DNN_y[i][j][0] = IncreaseData / maxPM

    def HighPMValueIncrease_v2(self, PMType='PM2.5'):
        badValue = 0
        maxPM = 0

        if PMType == "PM2.5":
            badValue = 35.5
            maxPM = 110.0
        elif PMType == "PM10":
            badValue = 80.5
            maxPM = 180.0

        for i in range(len(self.timelist)):
            for j in range(len(self.combineDate)):
                temp = round(self.DNN_y[i][j][0] * maxPM)
                standardPM = round(badValue - round(badValue * 0.1))
                # print("StandardPM : ", standardPM)
                if temp >= standardPM:
                    IncreaseData = (temp * 1.1)
                    if IncreaseData > maxPM:
                        IncreaseData = maxPM
                    self.DNN_y[i][j][0] = IncreaseData / maxPM


class Constant:
    def __init__(self, preStartEpoch=0, preMinDeltaCost=[0]):
        self.PRESTARTEPOCH = preStartEpoch
        # self.PREMAXDELTACOST = [1.0, 0.002, 0.002]
        self.PREMAXDELTACOST = [10, 10, 10]
        self.PREMINDELTACOST = copy.deepcopy(preMinDeltaCost)  # [0.0008, 0.0008, 0.0001]
        self.PRELEARNCOUNT = 0

        self.STARTEPOCH = 0
        # self.MAXDELTACOST = [0.003, 0.0015, 0.001]
        self.MAXDELTACOST = 0  # [10, 10, 10]
        self.MINDELTACOST = -1  # [0.01, 0.01, 0.01]


class WeightInit:
    def Xavier(self, data, numInput):
        xavierData = copy.deepcopy(data / np.sqrt(numInput))

        return xavierData

    def He(self, data, numInput):
        HeData = copy.deepcopy(data / np.sqrt(numInput / 2.0))

        return copy.deepcopy(HeData)