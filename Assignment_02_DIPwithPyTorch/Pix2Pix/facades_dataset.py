import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        if img_color_semantic is None:
            raise FileNotFoundError(f'Failed to read image: {img_name}')

        img_color_semantic = cv2.cvtColor(img_color_semantic, cv2.COLOR_BGR2RGB)
        height, width, _ = img_color_semantic.shape
        half_width = width // 2
        if half_width == 0:
            raise ValueError(f'Image width is too small to split into paired halves: {img_name}')

        color = img_color_semantic[:, :half_width]
        semantic = img_color_semantic[:, half_width:half_width * 2]

        # Resize paired images to the training resolution expected by the FCN.
        color = cv2.resize(color, (256, 256), interpolation=cv2.INTER_AREA)
        semantic = cv2.resize(semantic, (256, 256), interpolation=cv2.INTER_AREA)

        # Convert the image to a PyTorch tensor
        image_rgb = torch.from_numpy(color).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        image_semantic = torch.from_numpy(semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
        return image_rgb, image_semantic
