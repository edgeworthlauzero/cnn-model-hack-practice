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
        # 卷积层 3 -> 32 -> 64
        self.conv1 = nn.Conv2d(3, 32, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(32, 64, 5, pad_mode='valid')
        # 批量标准化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        # 最大池化层
        self.maxpool = nn.MaxPool2d(2, 2)
        # 展平层
        self.flatten = nn.Flatten()
        # 全连接层 1600 -> 256 -> 84 -> 10
        self.fc1 = nn.Dense(1600, 256)
        self.fc2 = nn.Dense(256, 84)
        self.fc3 = nn.Dense(84, 10)
    # 定义网络结构
    def construct(self, x):
        x = self.conv1(x) # 3 * 32 * 32 -> 32 * 28 * 28 
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 32 * 28 * 28 -> 32 * 14 * 14
        x = self.conv2(x) # 32 * 14 * 14 -> 64 * 10 * 10
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x) # 64 * 10 * 10 -> 64 * 5 * 5
        x = self.flatten(x) # 64 * 5 * 5 -> 1600
        x = self.fc1(x) # 1600 -> 256
        x = self.relu(x)
        x = self.fc2(x) # 256 -> 84
        x = self.relu(x)
        x = self.fc3(x) # 84 -> 10
        return x

# 加载数据集
def dataset(usage):
    # 加载数据集
    dataset = ds.Cifar10Dataset('datasets/cifar', usage=usage, shuffle=True)
    # 定义预处理操作
    if usage == 'train': # 若是训练集则进行数据增强
        image = [
            vision.RandomCrop(32, padding=4), # 随机裁剪填充
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
    dataset = dataset.map(operations=image, input_columns='image')
    dataset = dataset.map(operations=label, input_columns='label')
    # 设置批量处理大小
    dataset = dataset.batch(32)
    # 返回完成处理的数据集
    return dataset

# 加载权重
def weights():
    # 检查预训练权重文件
    if os.path.exists('weights/fgsm.ckpt'): # 若存在则直接加载
        weights = ms.load_checkpoint('weights/fgsm.ckpt')
        ms.load_param_into_net(Net, weights)
        return 1
    else: # 若不存在则进行训练
        losses = train()
        losscurve(losses)
        ms.save_checkpoint(Net, 'weights/fgsm.ckpt')
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
    for i in range(30):
        print(f'Epoch {i+1:>2}/30, Loss = {f'{sum(losses)/len(losses):.4f}' if losses else 'N/A'}')
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
def test():
    correct = 0
    total = 0
    # 逐批处理测试数据
    for image, label in Test_dataset.create_tuple_iterator():
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

# 实现 FGSM 攻击
def fgsm(image, label):
    # 初始化微分函数
    grad_fn = ms.grad(forward)
    # 计算损失值微分及其符号
    grads = grad_fn(image, label)
    grads_sign = ops.sign(grads)
    # 生成并处理对抗样本图像
    sample1 = image + (4 / 255) * grads_sign
    sample1 = ops.clamp(sample1, 0.0, 1.0)
    sample2 = image + (8 / 255) * grads_sign
    sample2 = ops.clamp(sample2, 0.0, 1.0)
    # 返回对抗样本图像
    return sample1, sample2

# 生成对抗样本
def perturb():
    count = 0
    results = []
    # 逐批处理测试数据
    for image, label in Test_dataset.create_tuple_iterator():
        if count >= 5:
            break
        # 获取原图像的预测值与标签
        pred = Net(image)
        real_pred = pred.argmax(axis=1).asnumpy()[0]
        real_label = label.asnumpy()[0]
        # 检查预测是否正确
        if real_pred != real_label: # 仅对预测正确的预测进行攻击
            continue
        count += 1
        # 获取对抗样本图像
        sample1, sample2 = fgsm(image, label)
        # 获取对抗样本图像的预测值
        sample1_pred = Net(sample1)
        sample2_pred = Net(sample2)
        real_sample1_pred = sample1_pred.argmax(axis=1).asnumpy()[0]
        real_sample2_pred = sample2_pred.argmax(axis=1).asnumpy()[0]
        # 保存结果
        results.append({
            'image': image,
            'sample1': sample1,
            'sample2': sample2,
            'real_label': real_label,
            'real_pred': real_pred,
            'real_sample1_pred': real_sample1_pred,
            'real_sample2_pred': real_sample2_pred
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
    plt.savefig('losscurves/fgsm.png')

# 保存对抗样本
def samples(results):
    # 检查对抗样本图像存储目录
    if not os.path.exists('samples/fgsm'): # 若不存在则创建
        os.makedirs('samples/fgsm')
    # 逐个处理结果
    for count, res in enumerate(results):
        # 提取对抗样本图像
        sample1 = res['sample1']
        sample2 = res['sample2']
        # 转换对抗样本图像数据
        sample1_display = sample1.asnumpy()[0].transpose(1, 2, 0)
        sample2_display = sample2.asnumpy()[0].transpose(1, 2, 0)
        # 保存对抗样本图像
        sample1_path = os.path.join('samples/fgsm', f'sample-{count+1}-1.png')
        sample2_path = os.path.join('samples/fgsm', f'sample-{count+1}-2.png')
        plt.imsave(sample1_path, sample1_display)
        plt.imsave(sample2_path, sample2_display)

# 绘制结果
def visualize(results):
    # 初始化画布
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    # 逐个处理结果
    for count, res in enumerate(results):
        # 提取结果
        image = res['image']
        sample1 = res['sample1']
        sample2 = res['sample2']
        real_label = res['real_label']
        real_pred = res['real_pred']
        real_sample1_pred = res['real_sample1_pred']
        real_sample2_pred = res['real_sample2_pred']
        # 转换原图像与对抗样本图像数据
        image_display = image.asnumpy()[0].transpose(1, 2, 0)
        sample1_display = sample1.asnumpy()[0].transpose(1, 2, 0)
        sample2_display = sample2.asnumpy()[0].transpose(1, 2, 0)
        # 计算扰动
        perturb1_display = sample1_display - image_display
        perturb2_display = sample2_display - image_display
        # 绘制原图像
        ax = axes[count][0]
        ax.imshow(image_display)
        ax.set_title(f'Image\nLabel: {Classes[real_label]}\nPred: {Classes[real_pred]}')
        ax.axis('off')
        # 绘制扰动
        ax = axes[count][1]
        normalized_perturb1_display = (perturb1_display - perturb1_display.min()) / (perturb1_display.max() - perturb1_display.min() + 1e-8)
        ax.imshow(normalized_perturb1_display)
        ax.set_title(f'\nPerturbation\n$\\epsilon$=4/255')
        ax.axis('off')
        ax = axes[count][2]
        normalized_perturb2_display = (perturb2_display - perturb2_display.min()) / (perturb2_display.max() - perturb2_display.min() + 1e-8)
        ax.imshow(normalized_perturb2_display)
        ax.set_title(f'\nPerturbation\n$\\epsilon$=8/255')
        ax.axis('off')
        # 绘制对抗样本图像
        ax = axes[count][3]
        ax.imshow(sample1_display)
        ax.set_title(f'Sample\n$\\epsilon$=4/255\nPred: {Classes[real_sample1_pred]}')
        ax.axis('off')
        ax = axes[count][4]
        ax.imshow(sample2_display)
        ax.set_title(f'Sample\n$\\epsilon$=8/255\nPred: {Classes[real_sample2_pred]}')
        ax.axis('off')
    # 设置布局
    fig.tight_layout()
    # 保存结果
    fig.savefig('results/fgsm.png')

if __name__ == '__main__':
    # 初始化模型
    Net = DNN()
    # 初始化优化器
    Optimizer = nn.SGD(
        Net.trainable_params(),
        learning_rate=1e-3,
        weight_decay=1e-4,
        momentum=0.9
    )
    # 初始化损失函数
    Loss = nn.CrossEntropyLoss()
    # 定义 CIFAR-10 类别标签
    Classes = ['airplane', 'automobile', 'bird', 'cat','deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    # 加载数据集
    print('\nFGSM hack')
    Train_dataset = dataset('train') # 训练集
    Test_dataset = dataset('test') # 测试集
    # 加载权重
    flag = weights()
    # 评估模型
    accuracy = test()
    if flag:
        print('\nUse pre-trained weights.')
    print(f'Model accuracy = {accuracy:.4f}')
    # 生成对抗样本图像
    results = perturb()
    # 保存对抗样本图像
    samples(results)
    print('\nSamples saved.')
    # 保存结果
    visualize(results)
    print('Results saved.')