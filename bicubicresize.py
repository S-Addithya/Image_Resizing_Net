import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Custom cubic interpolation

def cubic_interpolate(p, x):
    return (
        p[1] +
        0.5 * x * (p[2] - p[0] +
        x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] +
        x * (3.0 * (p[1] - p[2]) + p[3] - p[0])))
    )

def bicubic_interpolation(image, new_width, new_height):
    old_height, old_width, channels = image.shape
    new_image = np.zeros((new_height, new_width, channels))
    scale_x = old_width / new_width
    scale_y = old_height / new_height

    for k in range(channels):
        for new_y in range(new_height):
            for new_x in range(new_width):
                x = new_x * scale_x
                y = new_y * scale_y
                x0 = int(np.floor(x))
                y0 = int(np.floor(y))
                dx = x - x0
                dy = y - y0

                patch = np.zeros((4, 4))
                for m in range(-1, 3):
                    for n in range(-1, 3):
                        patch_x = np.clip(x0 + m, 0, old_width - 1)
                        patch_y = np.clip(y0 + n, 0, old_height - 1)
                        patch[m + 1, n + 1] = image[patch_y, patch_x, k]

                col0 = cubic_interpolate(patch[0, :], dx)
                col1 = cubic_interpolate(patch[1, :], dx)
                col2 = cubic_interpolate(patch[2, :], dx)
                col3 = cubic_interpolate(patch[3, :], dx)

                new_image[new_y, new_x, k] = cubic_interpolate([col0, col1, col2, col3], dy)

    return new_image

# Residual Block with Instance Normalization
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(filters)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.in1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.in2(x)
        x += residual
        x = self.leaky_relu(x)
        return x

# CNN Resizer with Bicubic Block Integration
class CNNResizer(nn.Module):
    def __init__(self, input_shape=(224, 224, 3), filters=16):
        super(CNNResizer, self).__init__()
        self.input_channels = input_shape[2]

        # Initial convolution layers
        self.conv1 = nn.Conv2d(self.input_channels, filters, kernel_size=7, padding=3)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=1)
        self.in1 = nn.InstanceNorm2d(filters)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(filters),
            ResidualBlock(filters),
            ResidualBlock(filters)
        )

        # Post residual block layers
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(filters)

        # Final convolution
        self.conv4 = nn.Conv2d(filters, 3, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initial convolutions
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.in1(x)

        # Residual blocks
        residual = x
        x = self.res_blocks(x)

        # Post-residual block processing
        x = self.conv3(x)
        x = self.in2(x)
        x += residual

        # Final convolution
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x

def process_images(input_dir, output_dir, new_width, new_height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)

            # Load the image
            input_image = Image.open(input_image_path)
            input_image = np.array(input_image)

            # Resize the image using bicubic interpolation
            resized_image = bicubic_interpolation(input_image, new_width, new_height)
            resized_image = np.clip(resized_image, 0, 255).astype(np.uint8)

            # Save the resized image
            resized_image_pil = Image.fromarray(resized_image)
            resized_image_pil.save(output_image_path)

# Example usage
if __name__ == "__main__":
    input_dir = r"D:\deep learning\sample12images"  # Replace with your input folder path
    output_dir = r"D:\deep learning\outresizecubic"  # Replace with your output folder path
    new_width = 224
    new_height = 224

    # Process the images in the dataset
    process_images(input_dir, output_dir, new_width, new_height)
