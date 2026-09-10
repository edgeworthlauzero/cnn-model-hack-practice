import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
import matplotlib.pyplot as plt
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
    # 定义中间输出
    def convolution(self, x):
        x = self.conv1(x) # 1 * 28 * 28 -> 6 * 24 * 24
        x = self.relu(x)
        x = self.maxpool(x) # 6 * 24 * 24 -> 6 * 12 * 12
        x = self.conv2(x) # 6 * 12 * 12 -> 16 * 8 * 8
        x = self.relu(x)
        x = self.maxpool(x) # 16 * 8 * 8 -> 16 * 4 * 4
        return x

# 加载数据集
def dataset(usage, flag):
    # 加载数据集
    dataset = ds.MnistDataset('datasets/mnist', usage=usage, shuffle=True)
    # 定义预处理操作
    if usage == 'train': # 若是训练集则进行数据增强
        image = [
            vision.RandomHorizontalFlip(), # 随机水平翻转
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.HWC2CHW()
        ]
    else: # 若是测试集或逆向集则只进行基本预处理
        image = [
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.HWC2CHW()
        ]
    label = transforms.TypeCast(ms.int32)
    # 应用预处理操作
    dataset = dataset.map(operations=image, input_columns='image')
    dataset = dataset.map(operations=label, input_columns='label')
    # 设置批量处理大小
    dataset = dataset.batch(1 if flag == 'invert' else 64)
    # 返回完成处理的数据集
    return dataset

# 加载权重
def weights():
    # 检查预训练权重文件
    if os.path.exists('weights/recst.ckpt'): # 若存在则直接加载
        weights = ms.load_checkpoint('weights/recst.ckpt')
        ms.load_param_into_net(Net, weights)
        return 1
    else: # 若不存在则进行训练
        losses = train()
        losscurve(losses)
        if not os.path.exists('weights'): # 若不存在则创建
            os.makedirs('weights')
        ms.save_checkpoint(Net, 'weights/recst.ckpt')
        return 0

# 训练时前向传播
def train_forward(image, label):
    # 获取预测值
    pred = Net(image)
    # 计算并返回损失
    loss = Train_loss(pred, label)
    return loss

# 逆向时前向传播
def invert_forward(sample, target):
    # 获取当前逆向样本对应的中间输出
    feature = Net.convolution(sample)
    # 计算损失
    loss = Invert_loss(feature, target)
    # 执行 L2 正则化
    l2 = Mean(sample ** 2)
    # 返回总损失
    return loss + 1e-3 * l2

# 训练模型
def train():
    losses = []
    # 开启训练模式
    Net.set_train(True)
    # 初始化微分函数
    grad = ms.value_and_grad(
        train_forward, # 前向传播函数
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

# 实现逆向恢复
def invert(image):
    # 初始化微分函数
    grad_fn = ms.grad(invert_forward)
    # 获取目标中间输出
    target = Net.convolution(image)
    # 初始化样本图像
    sample = ops.normal(
        shape=(1, 1, 28, 28),
        mean=0.1, stddev=0.2 
    )
    # 梯度优化迭代逆向样本图像
    for step in range(5000):
        # 计算损失值微分
        grads = grad_fn(sample, target)
        # 更新并处理逆向样本图像
        sample = sample - 0.01 * grads
        sample = ops.clamp(sample, 0.0, 1.0)
    # 返回逆向样本图像
    return sample

# 生成逆向样本
def reconstruct():
    count = 0
    results = []
    # 逐个处理数据
    for image, label in Target_dataset.create_tuple_iterator():
        if count >= 5:
            break
        # 获取原图像的预测值与标签
        pred = Net(image)
        real_pred = pred.argmax(axis=1).asnumpy()[0]
        real_label = label.asnumpy()[0]
        # 检查预测是否正确
        if real_pred != real_label: # 仅对预测正确的预测进行逆向恢复
            continue
        count += 1
        # 获取逆向样本图像
        sample = invert(image)
        # 获取逆向样本图像的预测值
        sample_pred = Net(sample)
        real_sample_pred = sample_pred.argmax(axis=1).asnumpy()[0]
        # 保存结果
        image = image.asnumpy().squeeze()
        sample = sample.asnumpy().squeeze()
        results.append({
            'image': image,
            'sample': sample,
            'real_label': real_label,
            'real_pred': real_pred,
            'real_sample_pred': real_sample_pred,
        })
    # 返回结果
    return results

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
    if not os.path.exists('losscurves'): # 若不存在则创建
        os.makedirs('losscurves')
    plt.savefig('losscurves/recst.png')

# 保存逆向样本
def samples(results):
    # 检查逆向样本图像存储目录
    if not os.path.exists('samples/recst'): # 若不存在则创建
        os.makedirs('samples/recst')
    # 逐个处理结果
    for count, res in enumerate(results):
        # 提取逆向样本图像
        sample_display = res['sample']
        # 保存逆向样本图像
        sample_path = os.path.join('samples/recst', f'sample-{count+1}.png')
        plt.imsave(sample_path, sample_display, cmap='gray')

# 绘制结果
def visualize(results):
    # 初始化画布
    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(15, 6))
    # 逐个处理结果
    for count, res in enumerate(results):
        # 提取结果
        image = res['image']
        sample = res['sample']
        real_label = res['real_label']
        real_pred = res['real_pred']
        real_sample_pred = res['real_sample_pred']
        # 绘制原图像
        ax = axes[0][count]
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Image\nLabel: {real_label}\nPred: {real_pred}')
        ax.axis('off')
        # 绘制逆向样本图像
        ax = axes[1][count]
        ax.imshow(sample, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Sample\nPred: {real_sample_pred}')
        ax.axis('off')
    # 设置布局
    fig.tight_layout()
    # 保存结果
    if not os.path.exists('results'): # 若不存在则创建
        os.makedirs('results')
    fig.savefig('results/recst.png')

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
    Train_loss = nn.CrossEntropyLoss()
    Invert_loss = nn.MSELoss()
    # 初始化求平均函数
    Mean = ops.ReduceMean()
    # 加载数据集
    print('\nReconstruction hack')
    Train_dataset = dataset('train', flag=None) # 训练集
    Test_dataset = dataset('test', flag=None) # 测试集
    Target_dataset = dataset('test', flag='invert') # 逆向集
    # 加载权重
    flag = weights()
    # 评估模型
    accuracy = test(Test_dataset)
    if flag:
        print('\nUse pre-trained weights.')
    print(f'Model accuracy on test data = {accuracy:.4f}')
    # 生成逆向样本图像
    results = reconstruct()
    # 保存逆向样本图像
    samples(results)
    print('\nSamples saved.')
    # 保存结果
    visualize(results)
    print('Results saved.')