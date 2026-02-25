import torch
import torch.nn as nn
import librosa
import numpy as np
import argparse
import math

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
BLANK_ID = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = logmel.T

    mean = logmel.mean(axis=0, keepdims=True)
    std = logmel.std(axis=0, keepdims=True) + 1e-8
    logmel = (logmel - mean) / std

    return logmel.astype(np.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        num_layers=6,
        num_heads=4,
        dim_ff=1024,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)
        x = self.positional_encoding(x)
        return self.encoder(x, src_key_padding_mask=mask)


class FrameClassifier(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.proj(x)


class FrameAlignmentModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.input_proj = nn.Linear(80, 256)

        self.encoder = TransformerEncoder(
            d_model=256,
            num_layers=6,
            num_heads=4,
            dim_ff=1024,
            dropout=0.1,
        )

        self.classifier = FrameClassifier(
            d_model=256,
            num_classes=num_classes,
        )

    def forward(self, x, lengths):
        x = self.input_proj(x)
        x = self.encoder(x, lengths)
        return self.classifier(x)


def extract_word_segments(preds):
    segments = []
    prev, start = BLANK_ID, None

    for i, p in enumerate(preds):
        if p != prev:
            if prev != BLANK_ID:
                segments.append((prev, start, i - 1))
            if p != BLANK_ID:
                start = i
            prev = p

    if prev != BLANK_ID:
        segments.append((prev, start, len(preds) - 1))

    return segments


def frame_to_time(frame):
    return frame * HOP_LENGTH / SAMPLE_RATE


def infer(audio_path, target_word):
    ckpt = torch.load("phrase_finders_model.pt", map_location=DEVICE)

    label_map = ckpt["label_map"]
    word_to_id = {k: int(v) for k, v in label_map.items()}
    id_to_word = {int(v): k for k, v in label_map.items()}

    if target_word not in word_to_id:
        print(f"Word '{target_word}' not in vocabulary")
        return

    features = extract_features(audio_path)
    T = features.shape[0]

    x = torch.tensor(features).unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([T]).to(DEVICE)

    model = FrameAlignmentModel(num_classes=len(label_map)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(x, lengths)
        preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

    segments = extract_word_segments(preds)
    target_id = word_to_id[target_word]

    found = False

    for sid, start_f, end_f in segments:
        if sid == target_id:
            print({
                "word": target_word,
                "start_time": round(frame_to_time(start_f), 3),
                "end_time": round(frame_to_time(end_f + 1), 3),
            })
            found = True

    if not found:
        print(f"Word '{target_word}' not found in audio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to mp3 file")
    parser.add_argument("--word", required=True, help="Target word")
    args = parser.parse_args()

    infer(args.audio, args.word)
