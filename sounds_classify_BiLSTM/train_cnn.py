from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchaudio import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from dataset import UrbanSound8k
from cnn import CNNNetwork

ANNOTATIONS_FILE = "data/UrbanSound8K_2.csv"
AUDIO_DIR = "data"
SAMPLE_RATE = 22050
NUM_SAMPLES = 32550 #决定 第三维度的数值 -> 1*64*64
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

#optimizer = optim.Adam(net.parameters(), lr = 0.01,  weight_decay=1e-3)
#optimizer = optim.rmsprop(net.parameters(), lr = 0.01, weight_decay=1e-3)
#optimizer = optim.RMSprop(net.parameters(), lr = 0.01,  weight_decay=1e-3)

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    return train_dataloader

# Training
def train(epoch,net,loaders):
    print('\nEpoch: %d' % epoch)
    net.train() # 表示 进入训练状态
    index = 0
    for i, (inputs, targets) in enumerate(loaders):
        index+=1
        inputs, targets = inputs.to(device), targets.to(device)
        b_x = Variable(inputs)
        b_y = Variable(targets)
        outputs = net(b_x)
        #outputs = net(inputs)
        loss = criterion(outputs, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        pass

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
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

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
    for i in range(int(train_len), total_len):
        test_dataset.append(usd[i])

    net = CNNNetwork().to(device)

    # state_dict = torch.load("cnn.pth")
    # net.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    trainloader = create_data_loader(train_dataset, BATCH_SIZE)
    testloader = create_data_loader(test_dataset, BATCH_SIZE)
    for epoch in range(100):
        train(epoch,net,trainloader)
        #test()
    torch.save(net.state_dict(), "cnn.pth")
    print("Trained feed forward net saved at cnn.pth")