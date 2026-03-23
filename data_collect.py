import cv2
import os
from pathlib import Path

# Определяем путь к папке "Видео" пользователя
video_folder = str(Path.home() / "videos")
output_path = os.path.join(video_folder, "barev0.mp4")

# Настройки захвата
cap = cv2.VideoCapture(0)

# Проверка, открылась ли камера
if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру")
    exit()

# Параметры видео (разрешение и частота кадров)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0  # Можно настроить под свою камеру

# Определяем кодек и создаем объект VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Запись началась. Видео будет сохранено в: {output_path}")
print("Нажмите 'q', чтобы остановить запись и сохранить файл.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Записываем кадр в файл
    out.write(frame)

    # Показываем окно с процессом записи
    cv2.imshow('Recording... Press Q to Save', frame)

    # Ждем нажатия клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

print("Запись завершена и сохранена.")