#!/usr/bin/env python3
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from attacks import Attacks
from variable import device

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024
batch_size = 32

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))


    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.

           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''
        self.load(os.path.join(project_dir, Net.model_file))



def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

def train_model_adversarial(net, train_loader, pth_filename, num_epochs, epsilon=0.3, alpha=0.01, num_steps=40):
    """Train the model with adversarial examples (adversarial training)."""
    print("Starting adversarial training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    attacks = Attacks(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs.requires_grad = True
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass the clean images through the model
            outputs_clean = net(inputs)
            loss_clean = criterion(outputs_clean, labels)

            # Compute gradients w.r.t. the clean images
            loss_clean.backward(retain_graph=True)
            data_grad = inputs.grad.data

            # Generate adversarial examples using FGSM and PGD
            perturbed_inputs_fgsm = attacks.fgsm_attack(inputs, epsilon, data_grad)
            perturbed_inputs_pgd = attacks.pgd_attack(net, inputs, labels, epsilon, alpha, num_steps)


            # Forward + backward + optimize for both FGSM and PGD examples
            outputs_fgsm = net(perturbed_inputs_fgsm)
            outputs_pgd = net(perturbed_inputs_pgd)

            # Compute the loss for both FGSM and PGD examples
            loss_fgsm = criterion(outputs_fgsm, labels)
            loss_pgd = criterion(outputs_pgd, labels)

            # Total loss is a combination of all
            loss = loss_fgsm + loss_pgd
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 500 == 499:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    net.save(pth_filename)
    print(f'Model saved in {pth_filename}')

def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def get_train_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid

def main():
    #### Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to store the model weights.")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists.")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training.")
    parser.add_argument('--epsilon', type=float, default=0.3,
                        help="Epsilon value for adversarial training (default is 0.3).")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Step size for PGD attack.")
    parser.add_argument('--num-steps', type=int, default=40,
                        help="Number of steps for PGD attack.")
    parser.add_argument('--adversarial', action="store_true",
                        help="Flag to enable adversarial training with FGSM and PGD.")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()])
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        print(next(iter(train_loader))[0].shape)
        # Train with combined adversarial examples (FGSM + PGD)
        if args.adversarial:
            train_model_adversarial(net, train_loader, args.model_file, args.num_epochs,
                                             epsilon=args.epsilon, alpha=args.alpha, num_steps=args.num_steps)
        else:
            train_model_regular(net, train_loader, args.model_file, args.num_epochs)

        print(f"Model saved to '{args.model_file}'.")

    #### Model testing
    print(f"Testing with model from '{args.model_file}'.")

    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())
    valid_loader = get_validation_loader(cifar, valid_size)

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print(f"Model natural accuracy (valid): {acc:.2f}")

if __name__ == "__main__":
    main()
