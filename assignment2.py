import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def create_dataloader():
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data',
                                               train=True,
                                               download=True,
                                               transform=transforms.ToTensor())

    test_dataset = torchvision.datasets.MNIST(root='data',
                                              train=False,
                                              download=True,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=50,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=50,
                                              shuffle=False)

    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=12, stride=2),
            nn.BatchNorm2d(25),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(25, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, num_classes)

        self.type = 'CNN'

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def train(train_loader, model, criterion, optimizer, epochs=4):
    # to train the model
    total_steps = len(train_loader)
    for e in range(epochs):
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # BP and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                correct = (outputs.argmax(1) == labels).sum().item()
                batchsize = labels.size(0)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy for this batch: {:.4f}'
                      .format(e + 1, epochs, step + 1, total_steps, loss.item(), correct / batchsize))


def test(test_loader, model, criterion):
    # to test the model

    correct = 0
    test_err = 0
    batches = len(test_loader)
    size = len(test_loader.dataset)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_err += loss.item()
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()

    correct /= size
    test_err /= batches
    # {test_err:>8f}
    print(f"The Accuracy of Test Set: {(100 * correct):>0.1f}%,\n Avg Loss:{test_err:>8f} \n")

def visualize_filter(model):
    model_children=list(model.children())
    #obtain first cnn layer filter with shape (25,1,12,12)
    filters=model_children[0][0].weight
    filters=filters.detach().cpu()

    plt.figure(figsize=(8,8))
    for idx, filtre in enumerate(filters):
        ax=plt.subplot(5,5,idx+1)

        ax.imshow(filtre.squeeze(),cmap="gray")
        ax.axis("off")
        ax.set_title("filter no.:{}".format(idx+1))
    plt.savefig('./filter.png')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### step 1: prepare dataset and create dataloader
    train_loader, test_loader = create_dataloader()

    ### step 2: instantiate CNN design model
    model = CNN()
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    ### step 3: train the model
    train(train_loader, model, criterion, optimizer, epochs=4)

    ### step 4: test the model
    test(test_loader, model, criterion)

    #step 5 visualize filters:
    visualize_filter(model)