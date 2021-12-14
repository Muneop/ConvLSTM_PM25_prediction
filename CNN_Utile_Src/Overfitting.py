import numpy as np
import copy
refValuePM25 = np.array((16, 36, 76))

"""

    타겟이 저농도 일때 예보가 고농도로 예보된 경우
    target < 36 and activation >= 36

"""

def CheckOverOutliers(target, activation, batchSize, batchIndex) :

    outliers = np.zeros(0, np.int)
    rightData = np.zeros(0, np.int)

    target = target * 110
    activation = activation * 110
    cnt = 0
    for i in range(batchSize) :
        if target[i] < refValuePM25[1] and activation[i] >= refValuePM25[1] :
            outliers = np.append(outliers, int(batchIndex[i]))
            cnt +=1
        else :
            rightData = np.append(rightData, int(batchIndex[i]))
    return rightData, outliers, cnt
"""

    타겟이 고농도 일때 예보가 저농도로 예보된 경우
    target >= 36 and activation < 36

"""
def CheckUnderOutliers(target, activation, batchSize, batchIndex) :
    outliers = np.zeros(0, np.int)
    rightData = np.zeros(0, np.int)

    target = target * 110
    activation = activation * 110
    cnt = 0
    for i in range(batchSize) :
        if target[i] >= refValuePM25[1] and activation[i] < refValuePM25[1] :
            outliers = np.append(outliers, int(batchIndex[i]))
            cnt +=1
        else :
            rightData = np.append(rightData, int(batchIndex[i]))
    return rightData, outliers, cnt

"""

    타겟이 고농도 일때 예보가 저농도로 예보된 경우
    target >= 36 and activation < 36
                or
    타겟이 저농도 일때 예보가 고농도로 예보된 경우
    target < 36 and activation >= 36
"""

def CheckAllOutliers(target, activation, batchSize, batchIndex) :
    outliers = np.zeros(0, np.int)
    rightData = np.zeros(0, np.int)

    target = target * 110
    activation = activation * 110
    cnt = 0

    for i in range(batchSize) :
        flag = False
        if round(target[i][0]) < refValuePM25[1] and round(activation[i][0]) >= refValuePM25[1] :
            flag = True

        elif round(target[i][0]) >= refValuePM25[1] and round(activation[i][0]) < refValuePM25[1] :
            flag = True

        if flag == True :
            outliers = np.append(outliers, int(batchIndex[i]))
            cnt += 1

        else :
            rightData = np.append(rightData, int(batchIndex[i]))

    return rightData, outliers, cnt



def RemoveOutliersData(obsTypeLength, obsTimeLength, dataLength, obsData, foreTypeLength, foreTime, foreData, flatData, targetData, rightData) :
    newObsData = np.zeros((obsTypeLength, obsTimeLength, len(rightData), dataLength), np.float)
    newforeData = np.zeros((foreTypeLength, 1,len(rightData), dataLength), np.float)
    newFlatData = np.zeros((1, len(rightData), flatData.shape[2]), np.float)
    newTarget = np.zeros((1, len(rightData), targetData.shape[2]), np.float)

    print(rightData)
    print(newObsData.shape)
    print(newforeData.shape)
    print(newFlatData.shape)
    print(newTarget.shape)

    print(obsData.shape)
    print(foreData.shape)
    print(flatData.shape)
    print(targetData.shape)

    ###### 측정 데이터
    for i in range(obsTypeLength) :
        for j in range(obsTimeLength) :
            for k in range(len(rightData)) :
                newObsData[i][j][k] = copy.deepcopy(obsData[i][j][rightData[k]])

    ###### 예보 데이터
        for i in range(foreTypeLength) :
            for j in range(len(rightData)) :
                newforeData[i][0][j] = copy.deepcopy(foreData[i][foreTime][rightData[j]])


    ###### DNN 데이터
    for i in range(len(rightData)) :
        newFlatData[0][i] = copy.deepcopy(flatData[foreTime][rightData[i]])
        newTarget[0][i] = copy.deepcopy(targetData[foreTime][rightData[i]])


    return copy.deepcopy(newObsData), copy.deepcopy(newforeData), copy.deepcopy(newFlatData), copy.deepcopy(newTarget)

def DataBalance(targetData, date, numCreateData = 1, PMType = 'PM2.5') :
    if PMType == "PM2.5":
        badValue = 35.5
        maxPM = 110.0
    elif PMType == "PM10":
        badValue = 80.5
        maxPM = 180.0

    print(PMType, badValue, maxPM)

    newDate = copy.deepcopy(date)
    dateTable = np.zeros((len(date)), np.int32)


    for i in range(len(date)) :
        dateTable[i] = i

    for i in range(len(date)) :
        originalData = round(targetData[i][0] * maxPM)
        if originalData >= badValue :
            for j in range(numCreateData) :
                newDate = np.append(newDate, date[i])
                dateTable = np.append(dateTable, i)

    print("number of Data %d -> %d" %(len(date), len(newDate)))

    return copy.deepcopy(newDate), copy.deepcopy(dateTable)