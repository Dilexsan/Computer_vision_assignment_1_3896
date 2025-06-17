import cv2
import matplotlib.pyplot as plt
from fun import (
    reduce_intensity_levels,
    spatial_average,
    rotate_image,
    block_average,
    test_code,
    
)

def main(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Task 1
    reduced_2 = reduce_intensity_levels(gray, 2)
    reduced_8 = reduce_intensity_levels(gray, 8)

    # Task 2
    avg_3 = spatial_average(gray, 3)
    avg_10 = spatial_average(gray, 10)
    avg_20 = spatial_average(gray, 20)

    # Task 3
    rot_45 = rotate_image(image, 45)
    rot_90 = rotate_image(image, 90)

    # Task 4
    block_3 = block_average(image, 3)
    block_5 = block_average(image, 5)
    block_7 = block_average(image, 7)

    # Show all results
    images = [
        ('Original Grayscale', gray),
        ('Reduced to 2 Levels', reduced_2),
        ('Reduced to 8 Levels', reduced_8),
        ('Average 3x3', avg_3),
        ('Average 10x10', avg_10),
        ('Average 20x20', avg_20),
        ('Rotated 45°', rot_45),
        ('Rotated 90°', rot_90),
        ('Block 3x3', block_3),
        ('Block 5x5', block_5),
        ('Block 7x7', block_7),
    ]

    for title, img in images:
        plt.figure()
        plt.title(title)
        plt.imshow(img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.show()

if __name__ == '__main__':
    path = f"D:\\d\\7\\EC7205 Computer Vision and Image Processing\\assignment1\\face.jpg"
    test_code()
    main(path)

    