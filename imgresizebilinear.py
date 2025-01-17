import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


# Custom Bilinear Resizer Block
class BilinearResizer(nn.Module):
    def __init__(self, target_height, target_width):
        super(BilinearResizer, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, x):
        # Custom bilinear interpolation logic
        n, c, hi, wi = x.shape  # Batch size, Channels, Height, Width
        output = torch.zeros((n, c, self.target_height, self.target_width), device=x.device, dtype=x.dtype)
        
        for i in range(n):  # Process each image in the batch
            for channel in range(c):  # Process each channel
                image = x[i, channel].cpu().detach().numpy()  # Convert to NumPy for processing
                resized_image = bilinear_resize(image, self.target_height, self.target_width)
                output[i, channel] = torch.tensor(resized_image, device=x.device)
        
        return output


# Bilinear interpolation function
def bilinear_resize(image, target_height, target_width):
    hi, wi = image.shape  # Original dimensions
    output_image = np.zeros((target_height, target_width), dtype=image.dtype)
    for y in range(target_height):
        for x in range(target_width):
            # Mapping output pixel to input pixel
            src_x = x / (target_width / wi)
            src_y = y / (target_height / hi)
            x0 = int(src_x)
            y0 = int(src_y)
            x1 = min(x0 + 1, wi - 1)
            y1 = min(y0 + 1, hi - 1)
            dx = src_x - x0
            dy = src_y - y0
            top_left = image[y0, x0]
            top_right = image[y0, x1]
            bottom_left = image[y1, x0]
            bottom_right = image[y1, x1]
            # Interpolating
            top = top_left * (1 - dx) + top_right * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx
            output_image[y, x] = top * (1 - dy) + bottom * dy
    return output_image


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.leaky_relu(x)
        return x


# CNN Resizer with Bilinear Block Integration
class CNNResizer(nn.Module):
    def __init__(self, input_shape=(224, 224, 3), filters=16):
        super(CNNResizer, self).__init__()
        self.input_channels = input_shape[2]

        # Custom bilinear resizer
        self.bilinear_resizer = BilinearResizer(224, 224)

        # Initial convolution layers
        self.conv1 = nn.Conv2d(self.input_channels, filters, kernel_size=7, padding=3)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm2d(filters)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(filters),
            ResidualBlock(filters),
            ResidualBlock(filters)
        )

        # Post residual block layers
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

        # Final convolution
        self.conv4 = nn.Conv2d(filters, 3, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Custom bilinear resize
        original = self.bilinear_resizer(x)

        # Initial convolutions
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn1(x)

        # Residual blocks
        residual = x
        x = self.res_blocks(x)

        # Post-residual block processing
        x = self.conv3(x)
        x = self.bn2(x)

        # First skip connection
        x += residual

        # Final convolution
        x = self.conv4(x)
        x = self.sigmoid(x)

        # Add resized original input
        x += original

        return x


# Process folder of images
def process_folder(input_folder, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        # Read image
        input_image = cv2.imread(os.path.join(input_folder, image_file))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        input_image_resized = cv2.resize(input_image, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        input_array = input_image_resized.astype(np.float32) / 255.0
        input_array = np.transpose(input_array, (2, 0, 1))  # CHW format
        input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Run through model
        output_tensor = model(input_tensor)
        output_array = output_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        output_array = np.clip(output_array, 0, 1) * 255  # Scale to 0-255
        output_image = output_array.astype(np.uint8)

        # Save the output
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, output_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Disable compression

    print(f"Processed {len(image_files)} images.")


# Example usage
if __name__ == "__main__":
    input_folder = "D:/deep learning/sample12images"
    output_folder = "D:/deep learning/outresize"
    model = CNNResizer(input_shape=(224, 224, 3), filters=16)
    process_folder(input_folder, output_folder, model)
