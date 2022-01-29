import torch
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from src.d2b2.train.HybridNN import HybridNN

train_dataset = dsets.MNIST(root="../../../data", train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataset = dsets.MNIST(root="../../../data", train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

width = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = HybridNN(width=width).to(device)
print(classifier)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

iteration = 0
epochs = 3
for epoch in range(epochs):
    print("epoch: ", epoch + 1)
    for i, (datapoints, labels) in enumerate(train_loader):
        datapoints = datapoints.reshape(-1, 28 * 28)
        optimizer.zero_grad()

        # We do a training loop that only differs from a common torch training
        # by having its loss sampled by its sample_elbo method.
        loss = classifier.sample_elbo(inputs=datapoints.to(device),
                                      labels=labels.to(device),
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1 / 50000)
        # print("train dataset loss: ", loss)
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 250 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.reshape(-1, 28 * 28)
                    outputs = classifier(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
            print('Iteration: {} | Accuracy of the network '
                  'on the 10000 test images: {} %'.format(str(iteration), str(100 * correct / total)))

PATH = './pth/HybridNN_d2b2_width%d_epochs%d.pth' % (width, epochs)

import os
if not os.path.exists(PATH):
    os.mknod(PATH)

torch.save(classifier.state_dict(), PATH)

print("over")
