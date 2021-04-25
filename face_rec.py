# модуль для работы с распознованием лиц
import cv2

# модуль для работы с изображениями
import matplotlib.pyplot as plt

# каскады Хаара (стандартные предустановленные каскады модуля)
face_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# открываем изображение
image = cv2.imread('faces2.jpg', 0)

# ищем лица по каскадам Хаара на изображении image
faces = face_cascade.detectMultiScale(
    image, # открытое ранее изображение
    scaleFactor=1.1, # шаг маштабирования 'скользящего окна'
    minNeighbors=5, # 'точность' распознования: чем меньше - тем более чувствителен поиск => больше ложных срабатываний (и наоборот)
)

# получаем начальные координаты и размеры лиц из массива
for (x, y, w, h) in faces: 
    
    # рисуем прямоугольинк по заданным координатам
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)


# отображаем изображение с распознанными лицами:

# задаём размер окна
plt.figure(figsize=(12, 12))

# отображаем в нём изоюражние с цветовым режимом (чёрно-белым)
plt.imshow(image, cmap='gray')

# отображаем само окно
plt.show()