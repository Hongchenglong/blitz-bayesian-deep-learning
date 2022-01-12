import os

from src.layer5.HybridNN import HybridNN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


test_dataset = dsets.MNIST(root="../data", train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


net = HybridNN()
PATH = '../../NNs/HybridNN_epoch1.pth'
pre_weights = torch.load(PATH)
net.load_state_dict(pre_weights)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.reshape(-1, 28 * 28)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))




