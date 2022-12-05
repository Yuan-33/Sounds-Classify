import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class UrbanSound8k(Dataset): #usd=UrbanSound8k() usd[1] ->get_item 返回值  signal 处理后的mel_spectrogram
    def __init__(self,
                 annotation, # label文件
                 audio_dir, # 音频文件位置
                 transform,
                 sample_rate, #sample时间内 采样率
                 num_samples, #sample 数量
                 device): #设备 "cuda"
        self.annotation = pd.read_csv(annotation,encoding="gbk")
        self.device = device
        self.audio_dir = audio_dir
        self.transform = transform.to(self.device)
        self.sample_rate = sample_rate
        self.num_samples = num_samples
    def __getitem__(self, item):
        path = self._get_audio_sample_path(item)
        label = self._get_audio_sample_label(item)  # 得到label
        signal, sr = torchaudio.load(path)  # signal wave图 sr 帧率
        signal = signal.to(self.device)  # -> gpu
        signal = self._resample_if_necessary(signal, sr)  # 转换为统一的帧率
        signal = self._mix_down_if_necessary(signal) #声道数可能不同
        signal = self._cut_if_necessary(signal)  # 若过长
        signal = self._right_pad_if_necessary(signal)  # 若过短，右边补0
        signal = self.transform(signal)
        return signal, label

    def __len__(self):
        return len(self.annotation) # csv文件长度

    #以下 非重写方法
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotation.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotation.iloc[
            index, 0])
        return path
    def get_label(self, index):
        return self.annotation.iloc[index, 6]
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            signal = resampler(signal).to(self.device)
        return signal
    def _get_audio_sample_label(self, index):
        return self.annotation.iloc[index, 6]

if __name__ == "__main__":
    ANNOTATIONS_FILE = "data/UrbanSound8K_2.csv"
    AUDIO_DIR = "data"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 32550 #决定 第三维度的数值 -> 64*64

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512, #n_fft 1/2
        n_mels=64
    )


    usd = UrbanSound8k(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    length = len(usd)
    print("音频文件个数:",length)
    for (image, label) in usd:
        print(image.shape)
        print(label)
    #未经过处理的音频文件 维度 [2,64,63] [2,64,751] [2,64,345] [1,64,376] -> 帧数不同 声道数也不同 （之前的测试）

