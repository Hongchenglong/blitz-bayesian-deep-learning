import os
import torch
from src.d2b2.train.HybridNN import HybridNN
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def mnist_test_point(filename):
    """
    获取mnist测试图片的784个像素点
    """
    x = []
    with open(filename) as f:
        for i in range(784):
            line = f.readline()
            x.append(float(line.strip()))
    return x

net = HybridNN()
PATH = './pth/HybridNN_d2b2_width64_epochs3.pth'
pre_weights = torch.load(PATH)
net.load_state_dict(pre_weights)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    images = mnist_test_point('../../mnist/mnist_2_local_property.in')
    labels = 7
    images = images.reshape(-1, 28 * 28)
    # calculate outputs by running images through the network
    outputs = net(images)
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))




