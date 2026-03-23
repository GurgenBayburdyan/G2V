import cv2
import os
from pathlib import Path

def get_unique_filename(folder, base_name, extension):
    """Generate a unique filename (barev.mp4, barev1.mp4, ...)."""
    counter = 0
    while True:
        suffix = "" if counter == 0 else str(counter)
        name = f"{base_name}{suffix}{extension}"
        full_path = os.path.join(folder, name)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

# Папка для видео
video_folder = str("videos")
os.makedirs(video_folder, exist_ok=True)

# Имя файла MP4
output_path = get_unique_filename(video_folder, "barev", ".mp4")

# Настройки камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Камера не найдена!")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0

# Кодек MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # для .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Запись идет в: {output_path}")
print("Нажмите 'q' на клавиатуре, чтобы остановить.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра.")
            break

        # Записываем кадр
        out.write(frame)

        # Показываем окно
        cv2.imshow('Recording... Press Q to Save', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Проверка файла
if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    print(f"Успешно сохранено! Размер файла: {os.path.getsize(output_path)} байт")
else:
    print("Ошибка: Файл не был создан или он пустой.")