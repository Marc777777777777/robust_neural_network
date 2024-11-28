#!/usr/bin/env python3

import os, os.path, sys
import argparse
import importlib
import importlib.abc
import torch, torchvision
import torchvision.transforms as transforms
from attacks import Attacks

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if os.path.exists(project_dir) and os.path.isdir(project_dir) and os.path.isfile(module_filename):
        print("Found valid project in '{}'.".format(project_dir))
    else:
        print("Fatal: '{}' is not a valid project directory.".format(project_dir))
        raise FileNotFoundError

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)

    return project_module

def test_natural(net, test_loader, num_samples):
    correct = 0
    total = 0
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            total = 0
            correct = 0
            for _ in range(num_samples):
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    return 100 * correct / total

def test_fgsm(net, test_loader, epsilon):
    attacks = Attacks(device)
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        # Compute gradients
        images.requires_grad = True
        outputs = net(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        net.zero_grad()
        loss.backward()
        data_grad = images.grad.data

        # Generate adversarial examples
        perturbed_images = attacks.fgsm_attack(images, epsilon, data_grad)

        # Test with adversarial examples
        outputs = net(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total


def test_pgd(net, test_loader, epsilon, alpha, steps):
    attacks = Attacks(device)
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        # Generate adversarial examples
        perturbed_images = attacks.pgd_attack(
            model=net,
            images=images,
            labels=labels,
            eps=epsilon,
            alpha=alpha,
            steps=steps,
        )

        # Test with adversarial examples
        outputs = net(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total

def test_cw(net, test_loader, c, kappa, steps, lr):
    attacks = Attacks(device)
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        # Generate adversarial examples
        perturbed_images = attacks.cw_attack(
            model=net,
            images=images,
            labels=labels,
            c=c,
            kappa=kappa,
            steps=steps,
            lr=lr,
        )
        # Test with adversarial examples
        outputs = net(perturbed_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100 * correct / total



def get_validation_loader(dataset, valid_size=1024, batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
    return valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", metavar="project-dir", nargs="?", default=os.getcwd(),
                        help="Path to the project directory to test.")
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Set batch size.")
    parser.add_argument("-s", "--num-samples", type=int, default=1,
                        help="Num samples for testing (required to test randomized networks).")
    parser.add_argument("--epsilon", type=float, default=0.3,
                        help="Epsilon for FGSM and PGD attacks.")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Step size for PGD attack.")
    parser.add_argument("--steps", type=int, default=3,
                        help="Number of steps for PGD attack.")
    parser.add_argument("--c", type=float, default=1e-4,
                        help="C value for Carlini-Wagner attack.")
    parser.add_argument("--kappa", type=float, default=0.01,
                        help="Kappa value for Carlini-Wagner attack.")
    parser.add_argument("--attack", nargs="+", choices=["pgd", "fgsm", "cw"], default=None,
                    help="Specify one or more attack types: 'pgd', 'fgsm'. If not provided, no attack is performed.")
    parser.add_argument("--param_noise", action="store_false", help="Disable parametric noise.")



    args = parser.parse_args()
    project_module = load_project(args.project_dir)

    net = project_module.Net(parametric_noise = args.param_noise)
    net.to(device)
    net.load_for_testing(project_dir=args.project_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
    valid_loader = get_validation_loader(cifar, batch_size=args.batch_size)


    if args.attack is None:
        print("No attack specified. Running natural accuracy test.")
        acc_nat = test_natural(net, valid_loader, num_samples=args.num_samples)
        print(f"Natural accuracy: {acc_nat:.2f}%")
    else:
        acc_nat = test_natural(net, valid_loader, num_samples=args.num_samples)
        print(f"Natural accuracy: {acc_nat:.2f}%")
        for attack in args.attack:
            if attack == "fgsm":
                acc_fgsm = test_fgsm(net, valid_loader, epsilon=args.epsilon)
                print(f"FGSM attack accuracy (epsilon={args.epsilon}): {acc_fgsm:.2f}%")
            elif attack == "pgd":
                acc_pgd = test_pgd(net, valid_loader, epsilon=args.epsilon, alpha=args.alpha, steps=args.steps)
                print(f"PGD attack accuracy (epsilon={args.epsilon}, alpha={args.alpha}, steps={args.steps}): {acc_pgd:.2f}%")
            elif attack == "cw":
                acc_cw = test_cw(net, valid_loader, c=args.c, kappa=args.kappa, steps=args.steps, lr=0.01)
                print(f"Carlini-Wagner attack accuracy (c=1e-4, kappa=0, steps=1000, lr=0.01): {acc_cw:.2f}%")
            else:
                print(f"Unknown attack: {attack}")



if __name__ == "__main__":
    main()
