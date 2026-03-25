import cv2
import os
import time

def get_unique_filename(folder, base_name, extension):
    counter = 0
    while True:
        suffix = "" if counter == 0 else str(counter)
        name = f"{base_name}{suffix}{extension}"
        full_path = os.path.join(folder, name)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

video_folder = "vidoes\\barev_0"
os.makedirs(video_folder, exist_ok=True)

output_path = get_unique_filename(video_folder, "barev", ".mp4")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Ошибка: Камера не найдена!")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 24.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"Запись идет в: {output_path}")
print("Запись будет длиться 5 секунд...")

start_time = time.time()
duration = 3

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра.")
            break

        out.write(frame)

        cv2.imshow('Recording... Press Q to Save', frame)

        if time.time() - start_time >= duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    print(f"Успешно сохранено! Размер файла: {os.path.getsize(output_path)} байт")
else:
    print("Ошибка: Файл не был создан или он пустой.")