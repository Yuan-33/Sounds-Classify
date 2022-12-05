from torch.utils.data import DataLoader
from torchaudio import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from lstm import Lstm
from dataset import UrbanSound8k

ANNOTATIONS_FILE = "data/UrbanSound8K_2.csv"
AUDIO_DIR = "data"
SAMPLE_RATE = 22050
NUM_SAMPLES = 32550 #决定 第三维度的数值 -> 1*64*64
BATCH_SIZE = 128

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device {device}")

net = Lstm(64, 100, 2, 10)

net = net.to(device)
state_dict = torch.load("lstm.pth")
net.load_state_dict(state_dict)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,weight_decay=0.003)
#optimizer = optim.Adam(net.parameters(), lr = 0.01,  weight_decay=1e-3)
#optimizer = optim.rmsprop(net.parameters(), lr = 0.01, weight_decay=1e-3)
optimizer = optim.RMSprop(net.parameters(), lr = 0.01,  weight_decay=1e-3)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    return train_dataloader
usd = UrbanSound8k(ANNOTATIONS_FILE,
                        AUDIO_DIR,
                        mel_spectrogram,
                        SAMPLE_RATE,
                        NUM_SAMPLES,
                        device)
total_len = len(usd)
train_len = total_len * 0.8
train_dataset = []
test_dataset = []
for i in range(int(train_len)):
    train_dataset.append(usd[i])
for i in range(int(train_len),total_len):
    test_dataset.append(usd[i])

trainloader = create_data_loader(train_dataset, BATCH_SIZE)
testloader = create_data_loader(test_dataset, BATCH_SIZE)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train() # 表示 进入训练状态
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(torch.squeeze(inputs, 1))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #print(targets)
        #print(predicted)
        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)
           # print(targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            #print(predicted)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



if __name__ == "__main__":
    for epoch in range(500):
        train(epoch)
        test()
    torch.save(net.state_dict(), "lstm.pth")
    print("Trained feed forward net saved at lstm.pth")
