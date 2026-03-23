import cv2
import os
from pathlib import Path

def get_unique_filename(folder, base_name, extension):
    counter = 0
    # Начинаем проверку с barev.avi, затем barev1.avi и т.д.
    while True:
        suffix = "" if counter == 0 else str(counter)
        name = f"{base_name}{suffix}{extension}"
        full_path = os.path.join(folder, name)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

# 1. Путь к папке "Видео"
video_folder = str(Path.home() / "videos")

# Создаем папку, если её вдруг нет
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# 2. Имя файла (используем .avi для надежности в Windows)
output_path = get_unique_filename(video_folder, "barev", ".avi")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Камера не найдена!")
    exit()

# 3. Настройки видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0

# Используем XVID — это самый стабильный кодек для Windows + OpenCV
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Запись идет в: {output_path}")
print("Нажмите 'q' (английскую) на клавиатуре, чтобы остановить.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра.")
            break

        # Записываем кадр
        out.write(frame)

        # Показываем окно
        cv2.imshow('Recording... Press Q to Save', frame)

        # ВАЖНО: Раскладка клавиатуры должна быть АНГЛИЙСКОЙ
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 4. Обязательно освобождаем ресурсы, иначе файл будет пустым (0 КБ)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    print(f"Успешно сохранено! Размер файла: {os.path.getsize(output_path)} байт")
else:
    print("Ошибка: Файл не был создан или он пустой.")