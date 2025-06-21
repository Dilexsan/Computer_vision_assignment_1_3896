import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def reduce_intensity_levels(image, levels):
    # Ensure levels is a power of 2
    assert (levels & (levels - 1) == 0) and levels > 0, "Levels must be a power of 2"
    factor = 256 // levels
    reduced = (image // factor) * factor
    return reduced.astype(np.uint8)

def average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def block_average(image, block_size):
    h, w = image.shape[:2]
    out = np.copy(image)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            avg = int(np.mean(block))
            out[i:i+block_size, j:j+block_size] = avg
    return out

def show_and_save_image(title, image, filename, cmap='gray'):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")

def main():
    image = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Image not found. Please make sure 'face.jpg' exists in the same directory.")
        return

    # Task 1: Reduce intensity levels (2 to 128)
    for levels in [2, 4, 8, 16, 32, 64, 128]:
        reduced = reduce_intensity_levels(image, levels)
        show_and_save_image(
            f'Intensity Reduced to {levels}', reduced, f'intensity_levels_{levels}.png'
        )

    # Task 2: Average filtering with different kernel sizes
    for k in [3, 10, 20]:
        avg = average_filter(image, k)
        show_and_save_image(
            f'{k}x{k} Average Filter', avg, f'average_filter_{k}x{k}.png'
        )

    # Task 3: Rotate the image
    rotated_45 = rotate_image(image, 45)
    rotated_90 = rotate_image(image, 90)
    show_and_save_image('Rotated 45 Degrees', rotated_45, 'rotated_45.png')
    show_and_save_image('Rotated 90 Degrees', rotated_90, 'rotated_90.png')

    # Task 4: Block averaging (3x3, 5x5, 7x7)
    for b in [3, 5, 7]:
        block_avg = block_average(image, b)
        show_and_save_image(
            f'Block Avg {b}x{b}', block_avg, f'block_average_{b}x{b}.png'
        )

if __name__ == "__main__":
    main()
