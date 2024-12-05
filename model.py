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
import matplotlib.pyplot as plt

from attacks import Attacks
from variable import device

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024
batch_size = 32


class RandomizedEnsembleClassifier(nn.Module):
  def __init__(self,classifiers,alpha):
    """
        Initialize the RCE

        Args:
            classifier (array): list of classifier.
            alpha (array): list of probabilities of each classifier.
        """
    super().__init__()
    self.classifiers = classifiers
    self.alpha = alpha

  def forward(self,x):
    """
        Forward pass of the RCE

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
    n_samples = x.shape[0]
    d = torch.multinomial(self.alpha.to(device), n_samples, replacement=True)
    return torch.cat([self.classifiers[d[i]](x[i:i+1]) for i in range(n_samples)])


class DeterministicEnsembleClassifier(nn.Module):
  def __init__(self,classifiers):
    """
        Initialize the DEC

        Args:
            classifier (array): list of classifier.
        """
    super().__init__()
    self.classifiers = classifiers

  def forward(self,x):
    """
        Forward pass of the DEC

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
    n_samples = x.shape[0]
    outputs = torch.stack([classifier(x) for classifier in self.classifiers], dim=0)
    return torch.sum(outputs,0)

class ParametricNoise(nn.Module):
    def __init__(self, shape, init_std=0.01):
        """
        Initialize learnable noise with the specified shape and standard deviation.

        Args:
            shape (tuple): Shape of the noise tensor.
            init_std (float): Initial standard deviation for noise initialization.
        """
        super().__init__()
        self.noise = nn.Parameter(torch.randn(shape) * init_std)

    def forward(self, x):
        """
        Add noise to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Input tensor with added noise.
        """
        return x + self.noise

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

    def __init__(self, parametric_noise=False):
        super().__init__()                        #Number of parameters:
        self.parametric_noise = parametric_noise
        self.conv1 = nn.Conv2d(3, 6, 5)           #5 x 5 × 3 × 6 + 6          = 456
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)          #5 x 5 × 6 × 16 + 16        = 2416
        self.fc1 = nn.Linear(16 * 5 * 5, 120)     #16 x 5 x 5 x 120 + 120     = 48120
        self.fc2 = nn.Linear(120, 84)             #120 x 84 + 120             = 10164
        self.fc3 = nn.Linear(84, 10)              #84 x 10 + 10               = 850
                                                  #Total                      = 61,006
        # Add parametric noise layers if enabled
        if parametric_noise:
            self.noise1 = ParametricNoise((6, 14, 14), init_std=0.01)  # After first pooling
            self.noise2 = ParametricNoise((16, 5, 5), init_std=0.01)  # After second pooling
            self.noise3 = ParametricNoise((16 * 5 * 5,), init_std=0.01)  # Before fully connected layers

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      #batch_size,6,28,28 then batch_size,6,14,14

        if self.parametric_noise:
            x = self.noise1(x)

        x = self.pool(F.relu(self.conv2(x)))      #batch_size,16,10,10 then batch_size,16,5,5

        if self.parametric_noise:
            x = self.noise2(x)

        x = torch.flatten(x, 1)                   #batch_size, 16*5*5

        if self.parametric_noise:
            x = self.noise3(x)

        x = F.relu(self.fc1(x))                   #batch_size, 120
        x = F.relu(self.fc2(x))                   #batch_size, 84
        x = self.fc3(x)                           #batch_size, 10
        x = F.log_softmax(x, dim=1)               #batch_size, 10
        return x



    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        directory = os.path.dirname(model_file)
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist
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

    def extract_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Fonction de similarité
def similarity(z1, z2):
    """Calculer la similarité cosinus entre deux vecteurs."""
    return F.cosine_similarity(z1, z2, dim=-1)

# Perte de contraste négative asymétrique
def asymmetric_negative_contrast(z_nat, z_neg, alpha=0.9):
    """
    Calcul de la perte de contraste négative asymétrique.
    Args:
        z_nat (torch.Tensor): Vecteurs de caractéristiques des exemples naturels.
        z_neg (torch.Tensor): Vecteurs de caractéristiques des exemples négatifs.
        alpha (float): Facteur de pondération pour l'asymétrie.
    Returns:
        torch.Tensor: La perte ANC.
    """
    sim_nat_neg = similarity(z_nat, z_neg)
    sim_neg_nat = similarity(z_neg, z_nat)
    loss_anc = alpha * sim_nat_neg + (1 - alpha) * sim_neg_nat.detach()
    return loss_anc.mean()

# Génération d'exemples négatifs
def generate_negatives(inputs, labels, model, epsilon=0.03, alpha=0.007, steps=10):
    """
    Génère des exemples négatifs à l'aide d'une attaque ciblée.
    Args:
        inputs (torch.Tensor): Exemples naturels.
        labels (torch.Tensor): Labels des exemples.
        model (torch.nn.Module): Modèle pour générer les attaques.
    Returns:
        torch.Tensor: Exemples négatifs.
    """
    targeted_labels = (labels + 1) % 10  # Exemple de label cible (prochaine classe)
    negatives = Attacks(device).pgd_attack(
        model, inputs, targeted_labels, eps=epsilon, alpha=alpha, steps=steps, targeted=True
    )
    return negatives


def train_model(net, train_loader, val_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)
    list_loss=[]
    list_val_loss = []
    accuracy_list = []
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
            if i % 500 == 499:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 500))
                list_loss.append(running_loss / 500)
                running_loss = 0.0

        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = net(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                _, predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        avg_val_loss = val_loss / len(val_loader)
        list_val_loss.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: Validation Loss: {avg_val_loss:.3f}")

    plt.plot(np.linspace(0,num_epochs,len(list_loss)+1)[1:],list_loss)
    plt.plot(range(1,num_epochs+1), list_val_loss, label="Validation Loss")
    plt.title("Evolution of Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.plot(range(1,num_epochs+1),accuracy_list, label="Accuracy")
    plt.title("Evolution of Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))


def train_model_adversarial(net, train_loader, val_loader, pth_filename, num_epochs, sigma=0.3, eps=0.03, alpha=0.01, steps=3, c=1e-4, kappa=0):
    """Train the model with adversarial examples (adversarial training)."""
    print("Starting adversarial training")
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    list_loss=[]
    list_val_loss = []
    accuracy_list = []
    attacks = Attacks(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # Randomly choose the attack type for this batch
            attack_type = torch.randint(0, 7, (1,)).item()  # 0: Smoothing, 1: FGSM, 2: PGD, 3: CW, Rest: None
            if attack_type == 0:
                # **Smoothing**: Add random Gaussian noise to inputs
                epsilon = torch.randn_like(inputs, device=device) * sigma
                inputs = torch.clamp(inputs + epsilon, 0, 1)
            elif attack_type == 1:
                # **FGSM Attack**: Create adversarial examples
                inputs.requires_grad = True
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                net.zero_grad()
                loss.backward()
                grad = inputs.grad.sign()
                inputs = attacks.fgsm_attack(inputs, eps, grad)
            elif attack_type == 2:
                # **PGD Attack**: Create adversarial examples
                inputs = attacks.pgd_attack(net, inputs, labels, eps, alpha, steps)
            elif attack_type == 3:
                # **Carlini-Wagner Attack**: Create adversarial examples
                inputs = attacks.cw_attack(net, inputs, labels, c= c , kappa= kappa, steps= steps, lr=0.01)
            else:
                # **No Attack**: Use the original inputs
                pass
            # Forward pass
            outputs = net(inputs)
            features = net.extract_features(inputs)
            loss_cls = criterion(outputs, labels)

            # Génération des exemples négatifs
            negatives = generate_negatives(inputs, labels, net)

            # Calcul de la perte ANC
            loss_anc = asymmetric_negative_contrast(features, net.extract_features(negatives))

            # Combinaison des pertes
            loss = loss_cls + 0.1 * loss_anc  # Pondération de la perte ANC

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 500 == 499:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 500:.3f}")
                list_loss.append(running_loss / 500)
                running_loss = 0.0

        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = net(val_inputs)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                _, predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        avg_val_loss = val_loss / len(val_loader)
        list_val_loss.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: Validation Loss: {avg_val_loss:.3f}, Validation Accuracy: {accuracy:.2f}")

    plt.plot(np.linspace(0,num_epochs,len(list_loss)+1)[1:],list_loss)
    plt.plot(range(1,num_epochs+1), list_val_loss, label="Validation Loss")
    plt.title("Evolution of Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.plot(range(1,num_epochs+1),accuracy_list, label="Accuracy")
    plt.title("Evolution of Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))



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
    parser.add_argument('-e', '--num-epochs', type=int, default=30,
                        help="Set the number of epochs during training.")
    parser.add_argument('--epsilon', type=float, default=0.3,
                        help="Epsilon value for adversarial training (default is 0.3).")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Step size for PGD attack.")
    parser.add_argument('--num-steps', type=int, default= 3,
                        help="Number of steps for PGD attack.")
    parser.add_argument('--adversarial', action="store_true",
                        help="Flag to enable adversarial training with FGSM and PGD.")
    parser.add_argument('--c', type=float, default=1e-4,
                        help="c parameter for CW attack.")
    parser.add_argument('--kappa', type=float, default=0.01,
                        help="kappa parameter for CW attack.")
    parser.add_argument("--param_noise", action="store_false", help="Disable parametric noise.")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net(parametric_noise=args.param_noise)
    net.to(device)
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        val_loader = get_validation_loader(cifar, valid_size, batch_size=batch_size)
        if args.adversarial:
            train_model_adversarial(net, train_loader, val_loader, args.model_file, args.num_epochs, eps=args.epsilon, alpha=args.alpha, steps=args.num_steps, c= args.c, kappa= args.kappa)
        else:
            train_model(net, train_loader, val_loader, args.model_file, args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print(f"Testing with model from '{args.model_file}'.")



    net.load(args.model_file)

    acc = test_natural(net, val_loader)
    print(f"Model natural accuracy (valid): {acc:.2f}")

if __name__ == "__main__":
    main()
