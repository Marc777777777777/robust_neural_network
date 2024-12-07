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

class LWTADense(nn.Module):
    def __init__(self, in_features, out_features, num_units):
        """
        Dense layer with LWTA activations.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_units = num_units
        self.linear = nn.Linear(in_features, out_features * num_units)

    def forward(self, x):
        """
        Forward pass for the dense LWTA layer.
        """
        x = self.linear(x)  # Shape: [batch_size, out_features * num_units]
        batch_size = x.size(0)
        x = x.view(batch_size, self.out_features, self.num_units)  # [batch_size, out_features, num_units]
        logits = F.softmax(x, dim=2)  # Probabilities
        winners = torch.multinomial(logits.view(-1, self.num_units), num_samples=1).squeeze(-1)  # Winner indices
        one_hot = F.one_hot(winners, num_classes=self.num_units).float()  # One-hot encoding
        one_hot = one_hot.view(batch_size, self.out_features, self.num_units)
        return (x * one_hot).sum(dim=2)  # Final output: [batch_size, out_features]

class LWTAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_units, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * num_units, kernel_size, stride, padding)
        self.num_units = num_units

    def forward(self, x):
        """
        Forward pass for LWTA convolutional layer.
        """
        x = self.conv(x)  # Shape: [batch_size, out_channels * num_units, height, width]
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, -1, self.num_units, height, width)  # Group units in blocks
        logits = F.softmax(x, dim=2)  # Compute probabilities
        logits_flat = logits.permute(0, 3, 4, 1, 2).reshape(-1, self.num_units)  # Flatten for sampling
        winners_flat = torch.multinomial(logits_flat, num_samples=1).squeeze(-1)  # Sample winners
        winners = winners_flat.view(batch_size, height, width, -1).permute(0, 3, 1, 2)  # Reshape winners
        one_hot = F.one_hot(winners, num_classes=self.num_units).permute(0, 1, 4, 2, 3).float()  # One-hot encoding
        return (x * one_hot).sum(dim=2)  # Final output: [batch_size, out_channels, height, width]


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

def predict_with_sampling(model, dataloader, num_samples=10):
    """
    Prédit les labels avec échantillonnage multiple.
    Args:
        model (nn.Module): Modèle avec LWTA.
        dataloader (torch.utils.data.DataLoader): Données à tester.
        num_samples (int): Nombre d'échantillons pour la moyenne.

    Returns:
        float: Précision moyenne.
    """
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_predictions = []

            for _ in range(num_samples):
                outputs = model(inputs)
                batch_predictions.append(outputs)

            # Moyenne des prédictions
            avg_predictions = torch.mean(torch.stack(batch_predictions), dim=0)
            _, predicted = avg_predictions.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

class LWTAELBOLoss(nn.Module):
    def __init__(self, temperature=1.0, num_units=2):
        """
        Initialise la perte ELBO pour les couches LWTA.
        Args:
            temperature (float): Facteur de température pour la relaxation.
            num_units (int): Nombre d'unités par bloc.
        """
        super(LWTAELBOLoss, self).__init__()
        self.temperature = temperature
        self.num_units = num_units
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, logits):
        """
        Calcule la perte ELBO.
        Args:
            outputs (torch.Tensor): Sorties du modèle (après softmax).
            targets (torch.Tensor): Labels réels.
            logits (torch.Tensor): Logits avant softmax.

        Returns:
            torch.Tensor: Perte ELBO combinée.
        """
        # Cross-Entropy Loss
        ce_loss = self.cross_entropy(outputs, targets)

        # KL Divergence
        log_q = F.log_softmax(logits / self.temperature, dim=-1)
        log_p = -torch.log(torch.tensor(self.num_units, device=logits.device).float())
        kl_divergence = (log_q - log_p).mean()

        # Combine les deux termes
        elbo_loss = ce_loss - kl_divergence
        return elbo_loss

class LWTAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_units, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * num_units, kernel_size, stride, padding)
        self.num_units = num_units

    def forward(self, x):
        x = self.conv(x)
        batch_size, _, height, width = x.size()
        x = x.view(batch_size, -1, self.num_units, height, width)
        logits = F.softmax(x, dim=2)
        logits_flat = logits.permute(0, 3, 4, 1, 2).reshape(-1, self.num_units)
        winners_flat = torch.multinomial(logits_flat, num_samples=1).squeeze(-1)
        winners = winners_flat.view(batch_size, height, width, -1).permute(0, 3, 1, 2)
        one_hot = F.one_hot(winners, num_classes=self.num_units).permute(0, 1, 4, 2, 3).float()
        return (x * one_hot).sum(dim=2)

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, widen_factor, stride=1):
        super().__init__()
        self.lwta_conv1 = LWTAConv(in_planes, planes * widen_factor, kernel_size=3, num_units=2, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes * widen_factor)
        self.lwta_conv2 = LWTAConv(planes * widen_factor, planes * widen_factor, kernel_size=3, num_units=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes * widen_factor)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * widen_factor:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * widen_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * widen_factor)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.lwta_conv1(x)))
        out = self.bn2(self.lwta_conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class Net(nn.Module):
    model_file="models/default_model.pth"
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''
    def __init__(self, widen_factor=1, num_classes=10):
        super().__init__()
        self.in_planes = 16

        def wide_layer(block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, widen_factor, stride))
                self.in_planes = planes * widen_factor
            return nn.Sequential(*layers)

        self.conv1 = LWTAConv(3, 16, kernel_size=3, num_units=2, stride=1, padding=1)
        self.layer1 = wide_layer(WideBasicBlock, 16, num_blocks=6, stride=1)
        self.layer2 = wide_layer(WideBasicBlock, 32, num_blocks=6, stride=2)
        self.layer3 = wide_layer(WideBasicBlock, 64, num_blocks=6, stride=2)
        self.bn = nn.BatchNorm2d(64 * widen_factor)
        self.fc = nn.Linear(64 * widen_factor, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

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
            if attack_type < 2:
                # **Smoothing**: Add random Gaussian noise to inputs
                epsilon = torch.randn_like(inputs, device=device) * sigma
                inputs = torch.clamp(inputs + epsilon, 0, 1)
            elif (attack_type < 4) and (attack_type >2 ):
                # **FGSM Attack**: Create adversarial examples
                inputs.requires_grad = True
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                net.zero_grad()
                loss.backward()
                grad = inputs.grad.sign()
                inputs = attacks.fgsm_attack(inputs, eps, grad)
            # elif attack_type == 2:
            #     # **PGD Attack**: Create adversarial examples
            #     inputs = attacks.pgd_attack(net, inputs, labels, eps, alpha, steps)
            # elif attack_type == 3:
            #     # **Carlini-Wagner Attack**: Create adversarial examples
            #     inputs = attacks.cw_attack(net, inputs, labels, c= c , kappa= kappa, steps= steps, lr=0.01)
            else:
                # **No Attack**: Use the original inputs
                pass
            # Forward pass with (potentially adversarial) inputs
            outputs = net(inputs)
            loss = criterion(outputs, labels)

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

        print(f"Epoch {epoch + 1}: Validation Loss: {avg_val_loss:.3f}")

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

def train_model_adversarial_with_elbo(net, train_loader, val_loader, pth_filename, num_epochs, temperature=1.0, epsilon=8/255, alpha=0.007, steps=20):
    """
    Entraîne le modèle avec adversarial training basé sur PGD en utilisant l'ELBO.
    """
    print("Starting adversarial training with PGD and ELBO")
    criterion = LWTAELBOLoss(temperature=temperature, num_units=2)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # Divise après 75 epochs
    attacks = Attacks(device)

    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Generate adversarial examples using PGD
            adv_inputs = attacks.pgd_attack(
                model=net, images=inputs, labels=labels,
                eps=epsilon, alpha=alpha, steps=steps
            )

            optimizer.zero_grad()

            # Forward pass uniquement avec des exemples adversariaux
            logits = net(adv_inputs)
            outputs = F.log_softmax(logits, dim=1)

            # Compute ELBO loss
            loss = criterion(outputs, labels, logits)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch >= 75:
            scheduler.step()

        # Validation
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_logits = net(val_inputs)
                val_outputs = F.log_softmax(val_logits, dim=1)
                val_loss += criterion(val_outputs, val_labels, val_logits).item()
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Enregistrement du modèle
    net.save(pth_filename)
    print(f"Model saved to {pth_filename}")

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
    net = Net(widen_factor=1)
    net.to(device)
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor())

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        val_loader = get_validation_loader(cifar, valid_size, batch_size=batch_size)
        if args.adversarial:
            train_model_adversarial_with_elbo(
        net, train_loader, val_loader,
        num_epochs=100, epsilon=8/255, alpha=0.007, steps=5, pth_filename=args.model_file
    )
        else:
            train_model(
                net,
                train_loader,
                val_loader,
                args.model_file,
                args.num_epochs
            )

        print(f"Model saved to '{args.model_file}'.")
    else:
        print(f"Model already exists at '{args.model_file}'. Skipping training.")


    #### Model testing
    print(f"Testing with model from '{args.model_file}'.")



    net.load(args.model_file)

    acc = test_natural(net, val_loader)
    print(f"Model natural accuracy (valid): {acc:.2f}")

if __name__ == "__main__":
    main()
