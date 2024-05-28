import cv2
import os

def convert_to_gray1(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(input_folder, filename))
            gray_img = img

            for i in range(gray_img.shape[0]):
                for j in range(gray_img.shape[1]):
                    gray_img[i, j] = min(255, max(0, round((img[i, j, 0] / 3 + img[i, j, 1] / 3 + img[i, j, 2] / 3))))

            cv2.imwrite(os.path.join(output_folder, filename), gray_img)

def convert_to_gray2(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(input_folder, filename))
            gray_img = img

            for i in range(gray_img.shape[0]):
                for j in range(gray_img.shape[1]):
                    gray_img[i, j] = int(0.299 * img[i, j, 2] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 0])

            cv2.imwrite(os.path.join(output_folder, filename), gray_img)

input_folder = "color_photos"
output_folder1 = "gray1"
output_folder2 = "gray2"

convert_to_gray1(input_folder, output_folder1)
convert_to_gray2(input_folder, output_folder2)
