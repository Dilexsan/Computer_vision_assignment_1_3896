import cv2
import numpy as np

def reduce_intensity_levels(image, levels):
    step = 256 // levels
    reduced = (image // step) * step
    return reduced

def spatial_average(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    return cv2.filter2D(image, -1, kernel)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def block_average(image, block_size):
    h, w = image.shape[:2]
    reduced_image = np.copy(image)

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = image[i:i+block_size, j:j+block_size]
            avg = np.mean(block, axis=(0, 1), dtype=np.uint8)
            reduced_image[i:i+block_size, j:j+block_size] = avg
    return reduced_image


def test_code():
    print("testing successfull")


if __name__ == "main":
    print("testing fun module")