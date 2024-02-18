import torch
import numpy as np
# from IPython import display
# import matplotlib.pylab as plt
# import torch.nn.functional as F
# import torch.optim as optim
import sys
import pandas as pd
# import sklearn.metrics
import random
# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import cross_val_score, KFold

sys.path.append("..")

print(torch.__version__)

#标准差标准化
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu) / std

def list_txt(path, list=None):
    '''

    :param path: 储存list的位置
    :param list: list数据
    :return: None/relist 当仅有path参数输入时为读取模式将txt读取为list
             当path参数和list都有输入时为保存模式将list保存为txt
    '''
    if list != None:
        file = open(path, 'w')
        # for item in list:
        #     for item1 in item:
        #         file.write(str(item1) + ' ')
        #     file.write('\n')
        for item in list:
            file.write(str(item))
            file.write('\n')
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

#Loss类
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        #mes_loss = 100 * torch.mean((torch.abs(x - y) / y).pow(2))
        #mes_loss = torch.mean((torch.abs(x - y) / y).pow(2))
        mse_loss = torch.mean((x - y).pow(2))
        return mse_loss
        #return torch.from_numpy(mes_loss)


#data = pd.read_excel('w17_18.xlsx')
data1 = pd.read_excel('move0.xlsx')

from sklearn.model_selection import train_test_split

# setup_seed(2021)
X, Y = np.array(data1[["Beta1", "Alpha1", "Beta2", "Alpha2"]], dtype='float32'), np.array(data1[["SPCF"]], dtype='float32')
np.random.seed(2024)
# input_train, res_train = np.array(data[["angle","area1","area2"]],dtype='float32'),np.array(data[["time"]],dtype='float32')
# X, Y = np.array(data1[["thick1", "angle1", "thick2"]], dtype='float32'), np.array(data1[["vol"]], dtype='float32')
# X, Y = np.array(data1[["thick1", "angle1", "thick2", "angle2"]], dtype='float32'), np.array(data1[["vol"]], dtype='float32')
# input_test, res_test = np.array(data1[["angle","area1","area2"]],dtype='float32'),np.array(data1[["time"]],dtype='float32')
# X_bias = 30*np.ones(len(Y),dtype='float32')

# from mayavi import mlab
# s = mlab.points3d(x,y,z)
# mlab.show()
# X = np.column_stack((X,X_bias))
# Y = Y / 100
# a_max = np.max(X[:, 1:3], axis=1)
# a_min = np.min(X[:, 1:3], axis=1)
# X[:, 1] = a_max
# X[:, 2] = a_min
# print('y_mean')
# print(Y.mean())
input_train, input_test, res_train, res_test = train_test_split(X, Y, test_size=0.2)


# input_train=np.append(input_train,input_train1,axis=0)
# res_train=np.append(res_train,res_train1,axis=0)
# input_train = feature_normalize(input_train)
# input_test = feature_normalize(input_test)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(4, 16)
        self.l2 = torch.nn.Linear(16, 16)
        self.l3 = torch.nn.Linear(16, 16)
        self.l4 = torch.nn.Linear(16, 16)
        self.l5 = torch.nn.Linear(16, 16)
        self.l6 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        x = torch.sigmoid(self.l5(x))
        return torch.relu(self.l6(x))


############################################ model = Net()
kf = KFold(n_splits=5)
models = []
# 设置损失函数和优化器
# criterion = torch.nn.MSELoss()
criterion = CustomLoss()
# 神经网络已经逐渐变大，需要设置冲量momentum=0.5

###########################################optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # ,momentum=0.9

writer = SummaryWriter("logs/loss_and_accuracy")
# 将一次迭代封装入函数中

# def train(epochs):
#     loss_list = []
#     loss_single = []
#     loss_list_test = []
#     last_loss = 0
#     min_loss = 1e9
#     for epoch in range(epochs):
#         model.train()
#         # 注意转行成tensor
#         inputs = torch.from_numpy(input_train)
#         labels = torch.from_numpy(res_train)
#         # 梯度要清零每一次迭代
#         optimizer.zero_grad()
#         # 前向传播
#         outputs: torch.Tensor = model(inputs)
#         # 计算损失
#         loss = criterion(outputs, labels)
#         if (loss < min_loss):
#             model1 = model
#             min_loss = loss
#         # 返向传播
#         loss.backward()
#         # 更新权重参数
#         optimizer.step()
#         if epoch % 5000 == 0:
#             print('epoch {}, loss {}'.format(epoch, loss.item()))
#             writer.add_scalar("loss/train", loss.item(), epoch)
#
#             model.eval()
#             with torch.no_grad():
#                 outputs_test: torch.Tensor = model(torch.from_numpy(input_test))
#                 labels_test = torch.from_numpy(res_test)
#                 loss_test = criterion(outputs_test, labels_test)
#                 writer.add_scalar("loss/test", loss_test.item(), epoch)
#     writer.close()
#     return model1

def train2(epochs):
    fold = 0
    for train_index, test_index in kf.split(X):
        input_train, input_test = X[train_index], X[test_index]
        res_train, res_test = Y[train_index], Y[test_index]

        train_columns = [f'feature_{i}' for i in range(input_train.shape[1])]
        test_columns = [f'feature_{i}' for i in range(input_test.shape[1])]

        train_df = pd.DataFrame(data=np.column_stack((input_train, res_train)), columns=train_columns+ ['target'])
        test_df = pd.DataFrame(data=np.column_stack((input_test, res_test)), columns=test_columns+ ['target'])

        train_df.to_csv(f'train_data_fold_{fold + 1}.csv', index=False)
        test_df.to_csv(f'test_data_fold_{fold + 1}.csv', index=False)


        min_loss = 1e9
        min_testloss = 1e9
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        testLoss = []
        trainLoss = []
        for epoch in range(epochs):
            model.train()
            # 注意转行成tensor
            inputs = torch.from_numpy(input_train)
            labels = torch.from_numpy(res_train)
            # 梯度要清零每一次迭代
            optimizer.zero_grad()
            # 前向传播
            outputs: torch.Tensor = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            if (loss < min_loss):
                model1 = model
                min_loss = loss
            # 返向传播
            loss.backward()
            # 更新权重参数
            optimizer.step()
            if (epoch % 5000 == 0) | (epoch == epochs - 1):
                print('epoch {}, loss {}'.format(epoch, loss.item()))
                # writer.add_scalar("loss/train", loss.item(), epoch)

                model.eval()
                with torch.no_grad():
                    outputs_test: torch.Tensor = model(torch.from_numpy(input_test))
                    labels_test = torch.from_numpy(res_test)
                    loss_test = criterion(outputs_test, labels_test)
                    test_loss = loss_test.item()
                    if (test_loss < min_testloss):
                        min_testloss = test_loss

            if (epoch % 500 == 0) | (epoch == epochs - 1):
                newlossitem = [epoch, loss.item()]
                trainLoss.append(newlossitem)

                model.eval()
                with torch.no_grad():
                    outputs_test: torch.Tensor = model(torch.from_numpy(input_test))
                    labels_test = torch.from_numpy(res_test)
                    loss_test = criterion(outputs_test, labels_test)
                    newtestlossitem = [epoch, loss_test.item()]
                    testLoss.append(newtestlossitem)
        #writer.close()

        trainLossdf = pd.DataFrame(trainLoss, columns=['ID','Loss'])
        testLossdf = pd.DataFrame(testLoss, columns=['ID', 'Loss'])

        trainLossdf.to_csv(f'train_Loss_fold_{fold + 1}.csv', index=False)
        testLossdf.to_csv(f'test_Loss_fold_{fold + 1}.csv', index=False)

        # 保存模型
        models.append(model1)

        print(f"Fold {fold + 1} - Train Loss: {min_loss:.3f}, Test Loss: {min_testloss:.3f}")
        fold += 1


def test():
    mean_test_loss = 0.0
    max_test_loss = 0.0
    PATH = "Fold_spcf/five_fold_1_spcf.pt"
    model = Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    size = len(X)
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    list = []
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for i in range(0, size):
            # 将数据转到GPU
            testX = X[i]
            testY = Y[i]
            # 将图片传入到模型当中就，得到预测的值pred
            input = torch.from_numpy(testX)
            pred = model(input)[0].item()
            newItem = [testY.item(), pred, (abs(testY.item() - pred) / pred) ]
            list.append(newItem)
            # 计算预测值pred和真实值y的差距
            # test_loss += criterion(pred, testY).item()
            # 统计预测正确的个数
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 返回相应维度的最大值的索引
    errorsea = pd.DataFrame(list, columns=['testY', 'pred', 'error'])
    errorsea.to_csv(f'errorspcf.csv', index=False)
    #list_txt('test100000_spcf.txt', list)
    #test_loss /= size
    #correct /= size
    #print('Accuracy on test set:%d %%' % (test_loss))

if __name__ == '__main__':
    test()
    # PATH = "move0_spcf_5000.pt"
    # model = train(100000)
    # torch.save(model.state_dict(), PATH)
    # torch.manual_seed(123)

            # indices = np.arange(len(X))
            # np.random.shuffle(indices)
            #
            # # 使用索引来重新排列X和y
            # X = X[indices]
            # Y = Y[indices]
            #
            # epochs = 50000
            # train2(epochs)
            # for i, model in enumerate(models):
            #     PATH = f'five_fold_{i + 1}_sea_{epochs}.pt'
            #     torch.save(model.state_dict(), PATH)

    # outputs: torch.Tensor = model(torch.from_numpy(input_train))
    #
    # err = sklearn.metrics.mean_absolute_error(res_train, outputs.detach().numpy())
    # print(err)
    # err = sklearn.metrics.mean_absolute_percentage_error(res_train, outputs.detach().numpy())
    # print(err)
    # outputs1: torch.Tensor = model(torch.from_numpy(input_test))
    # err = sklearn.metrics.mean_squared_error(res_test, outputs1.detach().numpy())
    # print(err)
    # err = sklearn.metrics.mean_absolute_error(res_test, outputs1.detach().numpy())
    # print(err)
    #
    # err = sklearn.metrics.mean_absolute_percentage_error(res_test, outputs1.detach().numpy())
    # print(err)
    # outputs1_np = outputs1.detach().numpy()

# Load
#    model = torch.load(PATH)
#    model.eval()