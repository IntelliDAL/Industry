import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from data.dataloader import task1_data,task2_data,task3_data,get_train_data,get_val_data,get_finaltest_data
#from __future__ import print_function, division
from model import share_lstm_model
import torch
import train
import pandas as pd
import utils

# 定义参数
class Arguments():
    def __init__(self,read_path = r"./data/CMAPSSData/",i_drop=0.1, h_drop=0.1,dropout=0.2,alpha=0.2, batch_size=128,test_batch_size=16,
                 epochs=200, lr=5e-4, weight_decay=5e-7,cnn_size = 128,mask_size = 7,
                 w_ent=1., nb_layers=3, h_size=32, seed=0, TIME_EMBED_SIZE=1, nb_measures=14, patience=15,d_model = 32,n_head=8,q_len =1,
                 n_kernels = 14,w_kernel = 1,task1_window_size = 30,task2_window_size = 60,task3_window_size = 90,residual_size=32, skip_size=32, dilation_cycles=2, dilation_depth=3,
                 validation="false", out="./output", verbose="True",sub_dataset="001",task1_sub_dataset="001_30",task2_sub_dataset="001_60",task3_sub_dataset="001_90",max_life=130,
                 scaler="mm",lradj = 'type1',scaler_range=(-1,1)):
        self.i_drop = i_drop
        self.h_drop = h_drop
        self.dropout = dropout
        self.alpha = alpha
        self.validation = validation
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.cnn_size = cnn_size
        self.mask_size = mask_size
        self.out = out
        self.w_ent = w_ent
        self.nb_layers = nb_layers
        self.h_size = h_size
        self.seed = seed
        self.verbose = verbose
        self.TIME_EMBED_SIZE = TIME_EMBED_SIZE
        self.nb_measures = nb_measures
        self.patience = patience
        self.d_model = d_model
        self.n_head = n_head
        self.n_kernels = n_kernels
        self.w_kernel = w_kernel
        self.q_len = q_len
        self.task1_window_size = task1_window_size
        self.task2_window_size = task2_window_size
        self.task3_window_size = task3_window_size
        self.residual_size=residual_size
        self.skip_size=skip_size
        self.dilation_cycles=dilation_cycles
        self.dilation_depth=dilation_depth
        self.sub_dataset=sub_dataset
        self.task1_sub_dataset=task1_sub_dataset
        self.task2_sub_dataset = task2_sub_dataset
        self.task3_sub_dataset = task3_sub_dataset
        self.max_life=max_life
        self.scaler=scaler
        self.lradj = lradj
        self.scaler_range=scaler_range
        self.batch_size=batch_size
        self.test_batch_size=test_batch_size
        self.read_path = read_path

def predict(model,finaltest_loader):
    model.eval()
    te_batch_dataset = []
    finaltest_output = []
    for i, (data_fte) in enumerate(finaltest_loader):
        temp = {}
        temp['task_name'] = 1
        temp['batch_data'] = data_fte
        te_batch_dataset.append(temp)

    for te_batchs in te_batch_dataset:  # 这里batchs是字典,batch_dataset=[temp1;temp2;temp3]
        te_batchs['batch_data']=te_batchs['batch_data'].float()

        te_batchs['batch_data'] = torch.Tensor(te_batchs['batch_data']).cuda(utils.CUDA)

        output_fte = model(te_batchs['batch_data'], te_batchs['task_name'])
        #output_fte += output_fte.cpu().flatten().tolist()
        finaltest_output += output_fte.cpu().flatten().tolist()

    return finaltest_output

#定义优化和损失
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred, actual))

if __name__ == '__main__':
    args = Arguments()
    #数据
    columns_drop=['setting1','setting2','setting3','s1','s5','s6','s10','s16','s18','s19']
    _,task1_train_loader = get_train_data(args.read_path, args.sub_dataset,args.task1_window_size,args.max_life, args.scaler,args.scaler_range,args.batch_size,columns_drop)
    _,task2_train_loader = get_train_data(args.read_path, args.sub_dataset, args.task2_window_size, args.max_life,args.scaler, args.scaler_range, args.batch_size, columns_drop)
    _,task3_train_loader = get_train_data(args.read_path, args.sub_dataset, args.task3_window_size, args.max_life,args.scaler, args.scaler_range, args.batch_size, columns_drop)

    _,task1_val_loader = get_val_data(args.read_path, args.sub_dataset,args.task1_window_size,args.max_life, args.scaler,args.scaler_range,args.batch_size,columns_drop)
    _,task2_val_loader = get_val_data(args.read_path, args.sub_dataset, args.task2_window_size, args.max_life,args.scaler, args.scaler_range, args.batch_size, columns_drop)
    _,task3_val_loader = get_val_data(args.read_path, args.sub_dataset, args.task3_window_size, args.max_life,args.scaler, args.scaler_range, args.batch_size, columns_drop)

    #_,task1_finaltest_loader = get_finaltest_data(args.read_path, args.task1_sub_dataset, args.task1_window_size, args.max_life, args.scaler, args.scaler_range,args.test_batch_size,columns_drop)
    _,task2_finaltest_loader = get_finaltest_data(args.read_path, args.task2_sub_dataset, args.task2_window_size,args.max_life, args.scaler, args.scaler_range, args.test_batch_size,columns_drop)
    _,task3_finaltest_loader = get_finaltest_data(args.read_path, args.task3_sub_dataset, args.task3_window_size,args.max_life, args.scaler, args.scaler_range, args.test_batch_size,columns_drop)

    # 损失
    criterion = RMSELoss()
    # 模型
    model = share_lstm_model.Sequence_Time_LSTM_1(args)
    # 训练
    train_loss_epoch, train_output = train.train(args, model, task1_train_loader, task2_train_loader,
                                                 task3_train_loader, task2_finaltest_loader, criterion)

    '''
    model.load_state_dict(torch.load('./checkpoint_cnn_wavenet_F_200.pth'))
    device = utils.device
    model.to(device)

    # 测试
    # 1类窗口的准确率和scores
    task1_pre_output = predict(model, task1_finaltest_loader)
    te_output = pd.DataFrame(task1_pre_output)
    te_output.to_csv('./30-40-50-result/test_output_cnn_wavenet_F_200.csv', index=None)
    true_rul = pd.read_csv(args.read_path + "RUL_FD" + args.task1_sub_dataset + ".csv")
    ture_rul = torch.FloatTensor(true_rul['RUL'].values)
    pre_rul = torch.FloatTensor(task1_pre_output)
    score = utils.score_func(pre_rul, ture_rul)
    RMSE = utils.RMSE(pre_rul, ture_rul)
    print("The task1Final Score is:", score.item())
    print("The task1Final RMSE is:", RMSE.item())
    '''
    '''
    # 2类窗口的准确率和scores
    task2_pre_output = predict(model, task2_finaltest_loader)
    te2_output = pd.DataFrame(task2_pre_output)
    te2_output.to_csv('./30_60_90_result/test_output_cnn_att_200_60.csv', index=None)
    true2_rul = pd.read_csv(args.read_path + "RUL_FD" + args.task2_sub_dataset + ".csv")
    ture2_rul = torch.FloatTensor(true2_rul['RUL'].values)
    pre2_rul = torch.FloatTensor(task2_pre_output)
    score2 = utils.score_func(pre2_rul, ture2_rul)
    RMSE2 = utils.RMSE(pre2_rul, ture2_rul)
    print("The task2Final Score is:", score2.item())
    print("The task2Final RMSE is:", RMSE2.item())

    # 3类窗口的准确率和scores
    task3_pre_output = predict(model, task3_finaltest_loader)
    te3_output = pd.DataFrame(task3_pre_output)
    te3_output.to_csv('./30_60_90_result/test_output_cnn_att_200_90.csv', index=None)
    true3_rul = pd.read_csv(args.read_path + "RUL_FD" + args.task3_sub_dataset + ".csv")
    ture3_rul = torch.FloatTensor(true3_rul['RUL'].values)
    pre3_rul = torch.FloatTensor(task3_pre_output)
    score3 = utils.score_func(pre3_rul, ture3_rul)
    RMSE3 = utils.RMSE(pre3_rul, ture3_rul)
    print("The task3Final Score is:", score3.item())
    print("The task3Final RMSE is:", RMSE3.item())
    '''


    '''
    task1_train_loader, task1_val_loader, task1_finaltest_loader = task1_data(args.read_path, args.sub_dataset,args.task1_window_size,
                                                                              args.max_life, args.scaler,args.scaler_range,
                                                                              args.batch_size,columns_drop)
    task2_train_loader, task2_val_loader, task2_finaltest_loader = task2_data(args.read_path, args.sub_dataset, args.task2_window_size,
                                                                              args.max_life, args.scaler, args.scaler_range,
                                                                              args.batch_size,columns_drop)
    task3_train_loader, task3_val_loader, task3_finaltest_loader = task3_data(args.read_path, args.sub_dataset, args.task3_window_size,
                                                                              args.max_life, args.scaler, args.scaler_range,
                                                                              args.batch_size,columns_drop)
    '''

    #测试
    #1类窗口的准确率和scores

    # 2类窗口的准确率和scores

    # 3类窗口的准确率和scores

    '''
    #保存训练输出
    tr_output = pd.DataFrame(train_output)
    tr_output.to_csv('./30_60_90_result/train_output_cnn_att_150.csv', index=None)
    #保存训练loss
    tr_loss = pd.DataFrame(train_loss_epoch)
    tr_loss.to_csv('./30_60_90_result/train_loss_cnn_att_150.csv', index=None)
    '''
    '''
    #任务一
    df = pd.DataFrame(train_output[0:17732])
    df.to_csv('./result/task1_train_output.csv',index = None)

    # 任务二
    df = pd.DataFrame(train_output[17732:34462])
    df.to_csv('./result/task2_train_output.csv', index=None)

    # 任务三
    df = pd.DataFrame(train_output[34462:50193])
    df.to_csv('./result/task3_train_output.csv', index=None)
    #任务一 train
    #task1_train_output = train_output[0:17731]
    #train_actual_predicted(read_path, sub_dataset, task1_window_size, max_life, task1_train_output, alpha_grid, alpha_low,alpha_high, _COLORS)
    #需要看任务一训练的100个发动机图，rmse和scores
    '''


