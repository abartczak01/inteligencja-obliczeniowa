import os
import cv2

bird_number = {
        "01.jpg": 3,
        "02.jpg": 7,
        "03.jpg": 6,
        "04.jpg": 2,
        "05.jpg": 20,
        "06.jpg": 5,
        "07.jpg": 1,
        "08.jpg": 1,
        "09.jpg": 2,
        "10.jpg": 2,
        "11.jpg": 2,
        "12.jpg": 4,
        "13.jpg": 9,
        "14.jpg": 11,
        "15.jpg": 23,
        "16.jpg": 2
}

count = 0
for filename in os.listdir("bird_miniatures"):
        gray_scaled = cv2.imread(os.path.join("bird_miniatures", filename), cv2.IMREAD_GRAYSCALE)

        img_contrast = cv2.convertScaleAbs(gray_scaled, alpha=2.7, beta=-250)
        # cv2.imwrite(os.path.join("test", "contrast_" + filename), img_contrast)

        edged = cv2.Canny(img_contrast, 30,150)
        # cv2.imwrite(os.path.join("test", "edged_" + filename), edged)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects_img = gray_scaled.copy()
        cv2.drawContours(objects_img, contours, -1, (0, 255, 0), 1)

        object_count = len(contours)
        cv2.imwrite(os.path.join("test", "objects_" + filename), objects_img)

        print(f"Na {filename} znaleziono: {object_count}, powinno być: {bird_number[filename]}")
        if object_count == bird_number[filename]:
                count+=1

print(f"liczba dobrze odgadniętych: {count}")
print(f"dokładność: {count/len(bird_number)*100} %")

