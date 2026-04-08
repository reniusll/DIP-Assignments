import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    num_images = min(num_images, inputs.shape[0])
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(image_rgb)

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, 'train_results', epoch)

        # Compute the loss
        loss = criterion(outputs, image_semantic)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a fully convolutional image-to-image translation model.')
    parser.add_argument('--train-list', default='train_list.txt', help='Path to the training image list.')
    parser.add_argument('--val-list', default='val_list.txt', help='Path to the validation image list.')
    parser.add_argument('--batch-size', type=int, default=100, help='Mini-batch size.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader worker processes.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--step-size', type=int, default=200, help='StepLR decay interval.')
    parser.add_argument('--gamma', type=float, default=0.2, help='StepLR decay factor.')
    parser.add_argument('--save-every', type=int, default=50, help='Checkpoint save interval.')
    return parser.parse_args()

def main():
    """
    Main function to set up the training and validation processes.
    """
    args = parse_args()

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file=args.train_list)
    val_dataset = FacadesDataset(list_file=args.val_list)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model, loss function, and optimizer
    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Training loop
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        validate(model, val_loader, criterion, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler.step()

        # Save model checkpoint every 50 epochs
        if (epoch + 1) % args.save_every == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
