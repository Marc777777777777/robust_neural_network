import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

class Attacks:
    def __init__(self, device):
        self.device = device

    # FGSM attack code
    def fgsm_attack(self, image, epsilon, data_grad):
        """
        Perform FGSM attack on the given image.

        Args:
            image (torch.Tensor): Original image.
            epsilon (float): Maximum perturbation.
            data_grad (torch.Tensor): Gradient of the loss w.r.t the input image.

        Returns:
            torch.Tensor: Adversarial image.

        """
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

    # PGD attack code
    def pgd_attack(self, model, images, labels, eps, alpha, steps, targeted=False, random_start=True):
        """
        Perform PGD attack on the given images.

        Args:
            model (nn.Module): The target model.
            images (torch.Tensor): Original images.
            labels (torch.Tensor): True labels for untargeted attack or target labels for targeted attack.
            eps (float): Maximum perturbation.
            alpha (float): Step size for each iteration.
            steps (int): Number of iterations.
            targeted (bool): Whether the attack is targeted.
            random_start (bool): Whether to start with a random perturbation.

        Returns:
            torch.Tensor: Adversarial examples.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if random_start:
            # Initialize with a random perturbation
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(steps):
            adv_images.requires_grad = True
            outputs = model(adv_images)

            # Calculate the loss
            if targeted:
                cost = -loss_fn(outputs, labels)  # Maximize the loss for targeted attack
            else:
                cost = loss_fn(outputs, labels)  # Minimize the loss for untargeted attack

            # Compute gradients
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            # Update adversarial images
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    # Restore the tensors to their original scale
    def denorm(self, batch, mean=[0.1307], std=[0.3081]):
        """
        Convert a batch of tensors to their original scale.

        Args:
            batch (torch.Tensor): Batch of normalized tensors.
            mean (torch.Tensor or list): Mean used for normalization.
            std (torch.Tensor or list): Standard deviation used for normalization.

        Returns:
            torch.Tensor: Batch of tensors without normalization applied to them.
        """
        if isinstance(mean, list):
            mean = torch.tensor(mean).to(self.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(self.device)

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


    def plot(self, images, adv_images, labels, adv_labels, title, mean=[0.1307], std=[0.3081], num_samples=5):
        """
        Plot a few samples of original images and their corresponding adversarial examples.

        Args:
            images (torch.Tensor): Original images.
            adv_images (torch.Tensor): Adversarial images.
            labels (torch.Tensor): True labels for the original images.
            adv_labels (torch.Tensor): Predicted labels for the adversarial images.
            mean (torch.Tensor or list): Mean used for normalization.
            std (torch.Tensor or list): Standard deviation used for normalization.
            num_samples (int): Number of samples to display.
        """

        # Denormalize the images
        images = self.denorm(images, mean, std)
        adv_images = self.denorm(adv_images, mean, std)

        # Ensure num_samples is not greater than the number of available images
        num_samples = min(num_samples, images.size(0))

        # Randomly select indices for the samples
        indices = random.sample(range(images.size(0)), num_samples)

        # Create the plot
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

        for i, idx in enumerate(indices):
            axes[0, i].imshow(images[idx].permute(1, 2, 0).detach().cpu().numpy().squeeze())
            axes[0, i].set_title(f"True: {labels[idx].item()}")
            axes[0, i].axis('off')

            axes[1, i].imshow(adv_images[idx].permute(1, 2, 0).detach().cpu().numpy().squeeze())
            axes[1, i].set_title(f"Adv: {adv_labels[idx].item()}")
            axes[1, i].axis('off')
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
