
import tensorflow as tf
import numpy as np
import datetime
import time
import copy
import threading
import sys, os


# \\220.66.59.120\ShareFolder\AS\003.CNN\001.AreaLearn\001.Module
# os.chdir("//220.66.59.120/ShareFolder/AS/003.CNN/001.AreaLearn/001.Module")
#
# now = os.getcwd()
# print(now)


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


from CNN_Utile_Src import DataInput3 as DI
from CNN_Utile_Src import Network3 as Network
from CNN_Utile_Src import Overfitting



# hyper parameters
print(tf.__version__)

# hyper parameters
# main Learn
learning_rate = 0.001
training_epochs = 10000
batch_size = 20
display_Step = 100#int(training_epochs / 100)

CONST = DI.Constant()
print(CONST.PRESTARTEPOCH)
#INDIM = 42#2021.06.29 MU // DNN 입력사이즈 42->186으로 증가, T6~T15 DNN병합때문, 원본
INDIM = 186#2021.06.29 MU // DNN 입력사이즈 42->186으로 증가, T6~T15 DNN병합때문, 수정

HIDDIM1 = 400
HIDDIM2 = 100
#OUTDIM = 1#2021.07.07 MU // DNN 출력 사이즈 변경 1->10(Tn -> T6~T15), 원본
OUTDIM = 10#2021.07.07 MU // DNN 출력 사이즈 변경 1->10(Tn -> T6~T15), 수정

ODataName = ["PM25", "U", "V", "TA", "RH"]
FDataName = ["PM25"]
#maxData = np.array([100, 310, 6, 5, 1.4, 0.08, 0.08, 180, 110, 0.01])
maxData = np.array([190, 8, 6, 35, 100])
maxData_f = np.array([200])
#minData = np.array([0, 260, -6, -5, 0, 0, 0, 0, 0, 0])
minData = np.array([0, -7, -3, -30, 5])
minData_f = np.array([0])
width = 146
height = 122
depth = 5
layerDim = [186, 4000, 1000, 1]#2021.07.17 MU //2DCNN 없이 실행한 훈련을 테스트 하기 위해 수정, 수정본
#layerDim = [186, 4000, 1000, 10]#2021.07.07 JM // DNN 입력사이즈 42->186으로 증가, T6~T15 DNN병합때문, 수정본

#layerDim = [42, 4000, 1000, 1]#2021.07.07 JM // DNN 입력사이즈 42->186으로 증가, T6~T15 DNN병합때문, 원본
divideMax = 110


#중부권 SU - 서울, IC - 인천, GGN - 경  기북부, GGS - 경기 남부, GWW -강원영서
#동남권 DG - 대구, GSN - 경북, GSS - 경남, BS -  부산,1 US - 울산
#서부권 DJ - 대전, SJ - 세종, CCS - 충남, CCN - 충북
#서남권 GJ - 광주, JLS - 전남, JLN - 전북
#영동 GWE
#제주 JJ
fcRegion = ["GWW"]
#00, 06, 12, 18 -> 03, 09, 15, 21
fcTime = "15"
timeDir = "15h/"
CNNInputPath = "../data/CNN/"

DNNInputPath=[]
for i in range(len(fcRegion)):
    DNNInputPath.append("../data/DNN/" + fcRegion[i] + "/")


ResultPath = "../result/"+ fcRegion[0] + "/"
# ResultPath = "../002.result/002.PM2.5/" + fcTime + "h/" + fcRegion + "/"


print(CNNInputPath)
print(DNNInputPath)
print(ResultPath)
if not os.path.exists(ResultPath):
    os.makedirs(ResultPath)
FileIndex = [i for i in range(6, 16, 1)]#2021.06.29 MU, T6~T15병합을 위해 삭제,
#FileIndex = [6]#2021.07.07 MU, T6~T15병합을 위해 삭제,

#FileIndex = [6, 8, 12]
#FileIndex = [7, 9, 10, 11, 13, 14, 15]
#FileIndex = [6]
obsTimeIndex = [i for i in range(1, 6)]



#PredictFileName = "./data/Predict_T06"
Gridex = ".ubyte"
Flatex = ".txt"

print("Start Predict File Open")

start_time = time.time()

predictData = DI.CNNDataset(timelist=FileIndex, obslist=ODataName, obsTimeList=obsTimeIndex, obsMax = maxData, obsMin = minData,
                          forelist=FDataName, foreTimeList=FileIndex, foreMax= maxData_f, foreMin = minData_f,
                          obs=True, fore = True, CNNPath = CNNInputPath, DNNPath=DNNInputPath, train=False, inputNum=INDIM, databalancing = False, numBalanceData = 0)
end_time = time.time()
times = str(datetime.timedelta(seconds=(end_time - start_time))).split(".")
times = times[0]
print(times)

print("End Predict File Open")

vgg16Network = Network.CNNNetwork(modelName="VGG16", numData=[len(FDataName), len(ODataName)], numLayer=3, inputDim= [depth, height, width], layerDim= layerDim, numGPUAuto = True, onlyCNN = False, preTraining = False)
vgg16Network.SettingNetwork(train=False)
vgg16Network.DisplayNetwork()



# initialize
sess = tf.Session()
saver = tf.train.Saver()

for i in [6]: # 2021.07.07 JM 수정
    with tf.Session() as sess:
        print("T06 ~ T15")
        ckpt_data = tf.train.get_checkpoint_state(ResultPath + "WGT/", latest_filename="T12_CheckPoint")
        print(ckpt_data, type(ckpt_data))
        weightFileNamae = ckpt_data.all_model_checkpoint_paths
        print(weightFileNamae, len(weightFileNamae))
        numPredict = len(weightFileNamae)

        for j in range(numPredict):
            print(j)
            print("WEIGHT NAME", weightFileNamae[j])
            saver.restore(sess, weightFileNamae[j])

            print("Finish Init")
            # train my model
            print("T06~T15" + "'Predict started. It takes sometime.")

            # Start Predict
            print("Start Predict")
            Pre_batch = 10
            Pre_cost = []
            Pre_hyp = []
            predictData.InitBatch()

            if predictData.len % Pre_batch == 0:
                Pre_total = int(predictData.len / Pre_batch)
            else:
                Pre_total = int(predictData.len / Pre_batch) + 1

            for Pre in range(Pre_total):

                batch_x, batch_y, batch_Grid_f, batch_Grid_o, batch_Index = predictData.NextBatchAll(Pre_batch, random=False, train = False)

                feed_dict = {}
                for inputIndex in range(len(ODataName)):
                    feed_dict[vgg16Network.inputData3D[inputIndex]] = batch_Grid_o[inputIndex]
                for inputIndex in range(len(FDataName)):
                    feed_dict[vgg16Network.inputData2D[inputIndex]] = batch_Grid_f[inputIndex]

                feed_dict[vgg16Network.DNNData] = batch_x
                feed_dict[vgg16Network.Y] = batch_y
                feed_dict[vgg16Network.dropoutProb] = 1.0

                C, hyp = sess.run([vgg16Network.cost, vgg16Network.hypothesis], feed_dict=feed_dict)


                Pre_cost.append(C)

                for Pre_result in range(len(batch_Index)):
                    Pre_hyp.append(hyp[Pre_result])

            #result = open(ResultPath + "Predict/" + "0_%03d_0090_T%02d" % (j, FileIndex[i]), "w")#2021.07.07 MU // 테스트 결과 T6~T15따로 저장
            result_6 = open(ResultPath + "Predict/" + "0_%03d_0090_T06.txt" % (j), "w")

            result_7 = open(ResultPath + "Predict/" + "0_%03d_0090_T07.txt" % (j), "w")
            result_8 = open(ResultPath + "Predict/" + "0_%03d_0090_T08.txt" % (j), "w")
            result_9 = open(ResultPath + "Predict/" + "0_%03d_0090_T09.txt" % (j), "w")
            result_10 = open(ResultPath + "Predict/" + "0_%03d_0090_T10.txt" % (j), "w")
            result_11 = open(ResultPath + "Predict/" + "0_%03d_0090_T11.txt" % (j), "w")
            result_12 = open(ResultPath + "Predict/" + "0_%03d_0090_T12.txt" % (j), "w")
            result_13 = open(ResultPath + "Predict/" + "0_%03d_0090_T13.txt" % (j), "w")
            result_14 = open(ResultPath + "Predict/" + "0_%03d_0090_T14.txt" % (j), "w")
            result_15 = open(ResultPath + "Predict/" + "0_%03d_0090_T15.txt" % (j), "w")

            print\
                ("PREDICTDATA LENGTH",predictData.len)
            for k in range(predictData.len):
                #print("몇번째에서 문제가 발생하는지 확인",k)
                #print("++++CASE4출력확인 시작++++")
                #print(predictData.DNN_x)
                #print(len(predictData.DNN_x[predictData.currentLearnTime][k]))
                temp = np.array(predictData.DNN_x)
                #print("shape확인",temp.shape)
                #print("predictData.DNN_x[predictData.currentLearnTime][k][14]",predictData.DNN_x[0][predictData.currentLearnTime][k][14])
                #print("++++끝++++")
                #result.write("%d\t" % k)
                '''20210707 MU // predict결과 저장 수정, 원본
                result.write("%7.2f\t" % (predictData.DNN_y[0][predictData.currentLearnTime][k][0] * divideMax))
                result.write("%7.2f\t" % (Pre_hyp[k][0] * divideMax))
                result.write("%7.2f\t" % (predictData.DNN_x[0][predictData.currentLearnTime][k][14] * divideMax))
                #[r][trainData.currentLearnTime][j][14]
                result.write("\n")
                '''

                #20210707 MU // predict결과 저장, 수정본 시작
                time_index = 0
                result_6.write("%7.2f\t " % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_6.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_6.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_6.write("\n")

                time_index = time_index + 1
                result_7.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_7.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_7.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_7.write("\n")

                time_index = time_index + 1
                result_8.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_8.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_8.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_8.write("\n")

                time_index = time_index + 1
                result_9.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_9.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_9.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_9.write("\n")

                time_index = time_index + 1
                result_10.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_10.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_10.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_10.write("\n")

                time_index = time_index + 1
                result_11.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_11.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_11.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_11.write("\n")

                time_index = time_index + 1
                result_12.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_12.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_12.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_12.write("\n")

                time_index = time_index + 1
                result_13.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_13.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_13.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_13.write("\n")

                time_index = time_index + 1
                result_14.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_14.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_14.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_14.write("\n")

                time_index = time_index + 1
                result_15.write("%7.2f\t" % (predictData.DNN_y[0][0][k][time_index] * divideMax))
                result_15.write("%7.2f\t" % (Pre_hyp[k][time_index] * divideMax))
                result_15.write("%7.2f\t" % (predictData.DNN_x[0][0][k][14 + time_index * 16] * divideMax))  # dnn인덱스 확인필요
                result_15.write("\n")



            #result.close()#20210707 MU // predict결과 저장 수정, 원본
            #20210707 JM // predict결과 저장 수정
            result_6.close()

            result_7.close()
            result_8.close()
            result_9.close()
            result_10.close()
            result_11.close()
            result_12.close()
            result_13.close()
            result_14.close()
            result_15.close()


            print("End Predict Test")
    predictData.EndLearn()
