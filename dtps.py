import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import matplotlib.pyplot as plt
import random as rd
import os

# 定义神经网络模型
class DNN(nn.Cell):
    # 定义神经元
    def __init__(self):
        super().__init__()
        # 激活函数
        self.relu = nn.ReLU()
        # 卷积层 1 -> 6 -> 16
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        # 最大池化层
        self.maxpool = nn.MaxPool2d(2, 2)
        # 展平层
        self.flatten = nn.Flatten()
        # 全连接层 256 -> 120 -> 84 -> 10
        self.fc1 = nn.Dense(256, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
    # 定义网络结构
    def construct(self, x):
        x = self.conv1(x) # 1 * 28 * 28 -> 6 * 24 * 24
        x = self.relu(x)
        x = self.maxpool(x) # 6 * 24 * 24 -> 6 * 12 * 12
        x = self.conv2(x) # 6 * 12 * 12 -> 16 * 8 * 8
        x = self.relu(x)
        x = self.maxpool(x) # 16 * 8 * 8 -> 16 * 4 * 4
        x = self.flatten(x) # 16 * 4 * 4 -> 256
        x = self.fc1(x) # 256 -> 120
        x = self.relu(x)
        x = self.fc2(x) # 120 -> 84
        x = self.relu(x)
        x = self.fc3(x) # 84 -> 10
        return x

# 加载数据集
def dataset(usage, flag):
    # 实现后门投毒
    def poison(image):
        img_poison = image.copy()
        img_poison[24:27, 24:27] = 255 # 后门图像右下角为白色方块
        label_poison = ms.Tensor(0) # 后门图像标签为 0
        return img_poison, label_poison
    def poisoner(image, label):
        if usage == 'train': # 若是训练集则按 10% 比例投毒
            return poison(image) if rd.random() < 0.1 else (image, label)
        elif flag == 'poison': # 若是后门测试集则全量投毒
            return poison(image)
        else: # 若是干净测试集则不投毒
            return image, label
    # 加载数据集
    dataset = ds.MnistDataset('datasets/mnist', usage=usage, shuffle=True)
    # 定义预处理操作
    if usage == 'train': # 若是训练集则进行数据增强
        image = [
            vision.RandomHorizontalFlip(), # 随机水平翻转
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.HWC2CHW() 
        ]
    else: # 若是测试集则只进行基本预处理
        image = [
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.HWC2CHW()
        ]
    label = transforms.TypeCast(ms.int32)
    # 应用预处理操作
    dataset = dataset.map(
        operations=poisoner,
        input_columns=['image', 'label']
    )
    dataset = dataset.map(operations=image, input_columns='image')
    dataset = dataset.map(operations=label, input_columns='label')
    # 设置批量处理大小
    dataset = dataset.batch(64)
    # 返回完成处理的数据集
    return dataset

# 加载权重
def weights():
    # 检查预训练权重文件
    if os.path.exists('weights/dtps.ckpt'): # 若存在则直接加载
        weights = ms.load_checkpoint('weights/dtps.ckpt')
        ms.load_param_into_net(Net, weights)
        return 1
    else: # 若不存在则进行训练
        losses = train()
        losscurve(losses)
        ms.save_checkpoint(Net, 'weights/dtps.ckpt')
        return 0

# 前向传播
def forward(image, label):
    # 获取预测值
    pred = Net(image)
    # 计算并返回损失
    loss = Loss(pred, label)
    return loss

# 训练模型
def train():
    losses = []
    # 开启训练模式
    Net.set_train(True)
    # 初始化微分函数
    grad = ms.value_and_grad(
        forward, # 前向传播函数
        grad_position=None, # 对权重进行微分
        weights=Optimizer.parameters
    )
    # 迭代训练
    print('\nTraining weights:')
    for i in range(20):
        print(f'Epoch {i+1:>2}/20, Loss = {f'{sum(losses)/len(losses):.4f}' if losses else 'N/A'}')
        # 逐批处理训练数据
        for image, label in Train_dataset.create_tuple_iterator():
            # 计算损失与梯度
            loss, grads = grad(image, label)
            # 记录损失
            losses.append(loss.asnumpy().item()) 
            # 更新权重
            Optimizer(grads)
    # 关闭训练模式
    Net.set_train(False)
    # 返回损失记录
    return losses

# 评估模型
def test(dataset):
    correct = 0
    total = 0
    # 逐批处理测试数据
    for image, label in dataset.create_tuple_iterator():
        # 获取预测值与标签
        pred = Net(image)
        real_pred = pred.argmax(axis=1).asnumpy()
        real_label = label.asnumpy()
        # 统计结果
        correct += (real_pred == real_label).sum()
        total += len(real_label)
    # 计算并返回准确率
    accuracy = correct / total
    return accuracy

# 绘制损失曲线
def losscurve(losses):
    # 初始化画布
    plt.figure(figsize=(10, 5))
    # 绘制损失曲线
    plt.plot(losses, label='Training Loss')
    # 设置标题与图表信息
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    # 保存损失曲线
    plt.savefig('losscurves/dtps.png')

if __name__ == '__main__':
    # 初始化模型
    Net = DNN()
    # 初始化优化器
    Optimizer = nn.SGD(
        Net.trainable_params(),
        learning_rate=1e-3,
        momentum=0.9
    )
    # 初始化损失函数
    Loss = nn.CrossEntropyLoss()
    # 加载数据集
    print('\nData poisoning hack')
    Train_dataset = dataset('train', None) # 训练集
    Test_Poisoned_dataset = dataset('test', 'poison') # 后门测试集
    Test_Unpoisoned_dataset = dataset('test', 'unpoison') # 干净测试集
    # 加载权重
    flag = weights()
    # 评估模型
    poisoned_accuracy = test(Test_Poisoned_dataset) # 后门测试集准确率
    unpoisoned_accuracy = test(Test_Unpoisoned_dataset) # 干净测试集准确率
    if flag:
        print('\nUse pre-trained weights.')
    print(f'Model accuracy on poisoned test data = {poisoned_accuracy:.4f}')
    print(f'Model accuracy on unpoisoned test data = {unpoisoned_accuracy:.4f}')