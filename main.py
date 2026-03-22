import cv2
import torch
import torch.nn as nn
import math
import numpy as np
import mediapipe as mp

# ======================
# MediaPipe setup
# ======================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

def extract_keypoints(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    keypoints = []

    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*33*3)

    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*21*3)

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*21*3)

    return np.array(keypoints)


def load_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        kp = extract_keypoints(frame)
        frames.append(kp)

    cap.release()

    # padding
    while len(frames) < max_frames:
        frames.append(np.zeros_like(frames[0]))

    return np.array(frames)


# ======================
# Transformer model
# ======================
class PositionalEncoding(nn.Module):
    def init(self, d_model, max_len=500):
        super().init()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class SignTransformer(nn.Module):
    def init(self, input_dim, num_classes):
        super().init()

        d_model = 128

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)

        x = x.mean(dim=1)
        return self.fc(x)


# ======================
# Dataset
# ======================
class VideoDataset(torch.utils.data.Dataset):
    def init(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def len(self):
        return len(self.video_paths)

    def getitem(self, idx):
        video = load_video(self.video_paths[idx])
        label = self.labels[idx]

        return torch.tensor(video, dtype=torch.float32), torch.tensor(label)


# ======================
# Example usage
# ======================
video_paths = [
    "video/barev.webm",
]

labels = [0, 1]

dataset = VideoDataset(video_paths, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

input_dim = 33*3 + 21*3 + 21*3  # pose + hands
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SignTransformer(input_dim, num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ======================
# Training
# ======================
for epoch in range(5):
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ======================
# Prediction
# ======================
model.eval()

video = load_video("video1.mp4")
video = torch.tensor(video, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(video)
    print("Prediction:", torch.argmax(pred, dim=1).item())