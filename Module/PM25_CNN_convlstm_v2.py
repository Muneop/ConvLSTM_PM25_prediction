import tensorflow as tfimport numpy as npimport datetimeimport timeimport copyimport sys, ostf.config.experimental_run_functions_eagerly(True)#tf.keras.backend.set_floatx('float32')gpus = tf.config.list_physical_devices('GPU')#tf.config.experimental.set_memory_growth(gpus[0], True)tf.config.experimental.set_virtual_device_configuration(gpus[0],        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])logical_gpus = tf.config.experimental.list_logical_devices('GPU')print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")#tf.compat.v1.disable_eager_execution()#tf.keras.backend.clear_session()sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))from CNN_Utile_Src import DataInput_convlstm_v2 as DIfrom CNN_Utile_Src import Network_convlstm_v2 as Networkfrom CNN_Utile_Src import Overfittingprint(tf.__version__)# hyper parameters# main Learnlearning_rate = 0.001 #0.009training_epochs = 2000batch_size = 1#batch_size = 1display_Step = 100 # 50 #int(training_epochs / 100)preLearn_epochs = 0prelearning_rate = 0.009Prebatch_size = 5Pre_display_Step = 100CONST = DI.Constant(preStartEpoch = 2000, preMinDeltaCost = 0.02)print(CONST.PRESTARTEPOCH, CONST.PREMINDELTACOST)#layerDim = [186, 4000, 1000, 400,1]#2021.07.07 JM // DNN 입력사이즈 42->186으로 증가, T6~T15 DNN병합때문, 수정본layerDim = [186, 4000, 1000, 1]#2021.07.07 JM // DNN 입력사이즈 42->186으로 증가, T6~T15 DNN병합때문, 수정본#layerDim = [186, 4000, 1000, 10]#2021.06.29 MU // DNN 입력사이즈 42->186으로 증가, T6~T15 DNN병합때문, 수정본#layerDim = [42, 4000, 1000, 1]#2021.06.29 MU // DNN 입력사이즈 42->186으로 증가, T6~T15 DNN병합때문, 원본# Grid ParametersODataName = ["PM25", "U", "V", "TA", "RH"]FDataName = ["PM25"]maxData = np.array([190, 8, 6, 35, 100])maxData_f = np.array([200])minData = np.array([0, -7, -3, -30, 5])minData_f = np.array([0])divideMax = 110randomSeed = 3np.random.seed(randomSeed)tf.random.set_seed(randomSeed)#중부권 SU - 서울, IC - 인천, GGN - 경  기북부, GGS - 경기 남부, GWW -강원영서#동남권 DG - 대구, GSN - 경북, GSS - 경남, BS -  부산,1 US - 울산#서부권 DJ - 대전, SJ - 세종, CCS - 충남, CCN - 충북#서남권 GJ - 광주, JLS - 전남, JLN - 전북#영동 GWE#제주 JJfcRegion = ["SU","IC","GGN","GGS","GWW"]#00, 06, 12, 18 -> 03, 09, 15, 21fcTime = "15"timeDir = "15h/"#for TrainDataCNNInputPath = "../data/CNN/"DNNInputPath=[]for i in range(0,len(fcRegion)):    DNNInputPath.append("../data/DNN/Learn/" + fcRegion[i] + "/")#for ValidationDataCNNInputPathVal = "../data/CNN/Validation/"DNNInputPathVal=[]for i in range(0,len(fcRegion)):    DNNInputPathVal.append("../data/DNN/Validation/" + fcRegion[i] + "/")ResultPath = "../result/"print(CNNInputPath)print(DNNInputPath)print(ResultPath)#집꿀벌#tnesorboardPath = "./board/log/"#FileIndex = [6]#2021.06.29 MU //  T6~T15를 하나로 묶기 위함FileIndex = [i for i in range(6, 16, 1)]#2021.06.29 MU // T6~T15를 하나로 묶기 위해 원본 삭제# FileIndex = [i for i in range(6, 16, 1)]#FileIndex = [6, 8, 12]#FileIndex = [6]obsTimeIndex = [i for i in range(1, 6)]width = 146height = 122depth = len(obsTimeIndex)print("Depth : %d, Height : %d, Width : %d" % (depth, height, width))#PredictFileName = "./data/Predict_T06"Gridex = ".ubyte"Flatex = ".txt"print("Start Learn File Open")start_time = time.time()inputDim = [depth,height,width]#hidden_dim = 15#기본 hidden dimhidden_dim = 5trainData = DI.CNNDataset(timelist=FileIndex, obslist=ODataName, obsTimeList=obsTimeIndex, obsMax = maxData, obsMin = minData,                          forelist=FDataName, foreTimeList=FileIndex, foreMax= maxData_f, foreMin = minData_f,                          obs=True, fore = True, CNNPath = CNNInputPath, DNNPath=DNNInputPath, train=True, inputNum=layerDim[0], databalancing = True, numBalanceData = 1,dataType = "PM2.5",lstmHiddenDim = hidden_dim)valData = DI.CNNDataset(timelist=FileIndex, obslist=ODataName, obsTimeList=obsTimeIndex, obsMax = maxData, obsMin = minData,                          forelist=FDataName, foreTimeList=FileIndex, foreMax= maxData_f, foreMin = minData_f,                          obs=True, fore = True, CNNPath = CNNInputPathVal, DNNPath=DNNInputPathVal, train=True, inputNum=layerDim[0], databalancing = True, numBalanceData = 1,dataType = "PM2.5",lstmHiddenDim = hidden_dim)end_time = time.time()times = str(datetime.timedelta(seconds=(end_time - start_time))).split(".")times = times[0]print(times)conv_lstm_model = Network.ConvLSTMVGG16_fixed(modelName="VGG16", numData=[len(FDataName), len(ODataName)], numLayer=3, inputDim= [depth, height, width], layerDim = layerDim, numGPUAuto = True, learnRate=learning_rate,batchsize =batch_size,hidden_dim=hidden_dim)ckpt_restore = True#model 선언#loss, optimizer 설정loss_object = tf.keras.losses.MeanSquaredError(name='mean_squared_error')#기존에는 RMSEloss 사용optimizer = tf.keras.optimizers.Adam()#기존에는 MomentumOptimizer 사용train_loss = tf.keras.metrics.Mean(name='train_loss')#기존에는 RMSEloss 사용test_loss = tf.keras.metrics.Mean(name='test_loss')#기존에는 RMSEloss 사용#CallBack Modelcheckpoint 옵션 설정ckpt_path = ResultPath+"/WGT/bestmodel"ckpt_dir = ResultPathcp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,save_weights_only=True,verbose=1)#train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')#test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')print("GPU avaliable:",tf.test.is_gpu_available())print("GPU name:",tf.test.gpu_device_name())#model = Network.ConvLSTMVGG16(modelName="VGG16", numData=[len(FDataName), len(ODataName)], numLayer=3, inputDim= [depth, height, width], layerDim = layerDim, numGPUAuto = True, learnRate=learning_rate,batchsize =batch_size,hidden_dim=hidden_dim)'''train_x, train_y, train_Grid_f, train_Grid_o, train_ecell, train_ehidden, train_dcell, train_dhidden,train_index = trainData.ReturnAllData(random=True, train=True)val_x, val_y, val_Grid_f, val_Grid_o, val_ecell, val_ehidden, val_dcell, val_dhidden, val_index = valData.ReturnAllData(random=True, train=False)inputs_cnn = tf.keras.Input(shape=(inputDim[0], inputDim[1] * inputDim[2],inputDim[0],))inputs_dnn = tf.keras.Input(shape=(layerDim[0],))#tf.placeholder(tf.float32, [None, self.layerDim[0]])inputs_ecell = tf.keras.Input(shape=(inputDim[1],inputDim[2],hidden_dim,))inputs_ehidden = tf.keras.Input(shape=(inputDim[1],inputDim[2],hidden_dim,))inputs_dcell = tf.keras.Input(shape=(inputDim[1],inputDim[2],hidden_dim,))inputs_dhidden = tf.keras.Input(shape=(inputDim[1],inputDim[2],hidden_dim,))outputs = Network.ConvLSTMVGG16(modelName="VGG16", numData=[len(FDataName), len(ODataName)], numLayer=3, inputDim= [depth, height, width], layerDim = layerDim, numGPUAuto = True, learnRate=learning_rate,batchsize =batch_size,hidden_dim=hidden_dim)(inputs_cnn,inputs_dnn,inputs_ecell,inputs_ehidden,inputs_dcell,inputs_dhidden)model = tf.keras.Model([inputs_cnn,inputs_dnn,inputs_ecell,inputs_ehidden,inputs_dcell,inputs_dhidden],outputs)model.compile(loss= tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.Adam())print(model.summary())model.fit([train_Grid_o,train_x,train_ecell,train_ehidden,train_dcell,train_dhidden],train_y,epochs=100,batch_size=5)##############################'''@tf.functiondef train_step(cnn_input,dnn_input,ecell, ehidden, dcell, dhidden, labels):    #추후에 batch_grid_f도 사용할 수 있도록 수정해야됨.    #cnn_input을 cnn_o_input, cnn_f_input으로 나눠야 됨    #batch_x, batch_y, batch_Grid_f, batch_Grid_o, _, _, _, _, batch_index = trainData.NextBatchAll(batch_size, random=True, train=True)    with tf.GradientTape() as tape:        # training=True is only needed if there are layers with different        # behavior during training versus inference (e.g. Dropout).        #predictions = conv_lstm_model([cnn_input,dnn_input,ecell,ehidden,dcell,dhidden], training=True)        predictions = conv_lstm_model(cnn_input,dnn_input,ecell,ehidden,dcell,dhidden, training=True)        loss = loss_object(labels, predictions)        loss += sum(conv_lstm_model.losses)    gradients = tape.gradient(loss, conv_lstm_model.trainable_variables)    optimizer.apply_gradients(zip(gradients, conv_lstm_model.trainable_variables))    train_loss(loss)@tf.functiondef val_step(cnn_input,dnn_input,ecell, ehidden, dcell, dhidden, labels):    #predictions = conv_lstm_model([cnn_input,dnn_input,ecell,ehidden,dcell,dhidden], training=False)    predictions = conv_lstm_model(cnn_input,dnn_input,ecell,ehidden,dcell,dhidden, training=False)    #print("len(dnn_input)",len(dnn_input))    val_loss = loss_object(labels, predictions)    test_loss(val_loss)    #print("labels, predictinos",labels, predictions)    return [labels,predictions]#T6~T15까지 한번에 학습@tf.functiondef main():    start_time = time.time()    f = open(ResultPath+"SelfTest/"+"T06~T15_CONVLSTM.txt","w")    minimum_val_loss = 100    for epoch in range(training_epochs + 1):        avg_cost = 0        cost_sum = 0        crubatch = 0        train_loss.reset_states()        test_loss.reset_states()        trainData.InitBatch()        if trainData.len % batch_size == 0:            total_batch = int(trainData.len / batch_size)        else:            total_batch = int(trainData.len / batch_size) + 1        for pattern in range(total_batch):            batch_x, batch_y, batch_Grid_f, batch_Grid_o, batch_ecell, batch_ehidden, batch_dcell, batch_dhidden,batch_index = trainData.NextBatchAll(                batch_size, random=True, train=True)            #batch_Grid_f는 T6~T15 예보 데이터, 추후에 사용할 수 있도록 미리 변수 할당            train_step(batch_Grid_o,batch_x,batch_ecell, batch_ehidden, batch_dcell, batch_dhidden, batch_y)        val_result = []        valData.InitBatch()        val_length = len(valData.combineDate)        if valData.len % batch_size == 0:            total_batch_val = int(valData.len / batch_size)        else:            total_batch_val = int(trainData.len / batch_size) + 1        for pattern in range(total_batch_val):            val_batch_x, val_batch_y, val_batch_Grid_f, val_batch_Grid_o, val_batch_ecell, val_batch_ehidden, val_batch_dcell, val_batch_dhidden,val_batch_index = valData.NextBatchAll(                batch_size, random=True, train=False)            #batch_Grid_f는 T6~T15 예보 데이터, 추후에 사용할 수 있도록 미리 변수 할당            val_label, val_prediction = val_step(val_batch_Grid_o,val_batch_x,val_batch_ecell, val_batch_ehidden, val_batch_dcell, val_batch_dhidden, val_batch_y)            for i in range(len(val_batch_index)):                val_result.append(val_prediction[i])        #val_case24 = valData.DNN_x[r][0][j][14+time_index*16]*divideMax        if test_loss.result()<minimum_val_loss:            minimum_val_loss = test_loss.result().numpy()            conv_lstm_model.save_weights(ckpt_path)            #val_result_txt = [open(ResultPath+"SelfTest/validation_"+str(epoch)+"_T"+str(i+6)+".txt") for i in range(10)]            val_result_txt = []            for i in range(10):                val_result_txt.append(open(ResultPath+"SelfTest/validation_"+str(epoch)+"_T"+str(i+6)+".txt","w"))            for r in range(len(fcRegion)):                for i in range(10):                    val_result_txt[i].write(fcRegion[r]+"\n")                for j in range(val_length):                    for time_index in range(10):                        #print(len(val_prediction))                        #print(val_prediction)                        val_result_txt[time_index].write("%7.2f\t" % (valData.DNN_y[r][0][j][time_index] * divideMax))                        val_result_txt[time_index].write("%7.2f\t" % (val_result[j][time_index] * divideMax))                        val_result_txt[time_index].write("%7.2f\t" % (valData.DNN_x[r][0][j][14 + time_index * 16] * divideMax))                        val_result_txt[time_index].write("\n")            for i in range(10):                val_result_txt[i].close()        print(            f'Epoch {epoch + 1}, '            f'Train Loss: {train_loss.result().numpy()}, '            f'Validation Loss: {test_loss.result().numpy()}, '        )        f.write("Epoch:"+str(epoch)+", Train Loss:"+str(train_loss.result().numpy())+", Validation Loss:"+str(test_loss.result().numpy())+"\n")    end_time = time.time()    times = str(datetime.timedelta(seconds=(end_time - start_time))).split(".")    times = times[0]    print(times)    f.write((times[0]))    f.close()if __name__=='__main__':    main()