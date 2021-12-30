import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator

train_dataset = dsets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataset = dsets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


@variational_estimator
class HybridNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = BayesianLinear(64, 10)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = HybridNN().to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

iteration = 0
for epoch in range(2):
    print("epoch: ", epoch + 1)
    for i, (datapoints, labels) in enumerate(train_loader):
        datapoints = datapoints.reshape(-1, 28 * 28)
        optimizer.zero_grad()
        loss = classifier.sample_elbo(inputs=datapoints.to(device),
                                      labels=labels.to(device),
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1 / 50000)
        # print(loss)
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 250 == 0:
            print("loss: ", loss)
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

print("over")