import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from ucimlrepo import fetch_ucirepo
import shap
from deeplift.visualization import viz_sequence
import matplotlib.pyplot as plt
from deeplift.dinuc_shuffle import dinuc_shuffle

# CNN model
class SpliceCNN(nn.Module):
    def __init__(self, input_size, num_classes, kernel_size=5):
        super(SpliceCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 8, kernel_size=kernel_size, padding=0)
        self.fc1 = nn.Linear(8 * (input_size - kernel_size + 1), 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        return x


# nucleotide dataset class
class SpliceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        return seq, label


# one-hot DNA encoder
def encode_dna(seq):
    encoding = {'A': [1,0,0,0], 
                'C': [0,1,0,0], 
                'G': [0,0,1,0], 
                'T': [0,0,0,1],
                'D': [1/3,0,1/3,1/3],
                'N': [1/4,1/4,1/4,1/4],
                'S': [0,1/2,1/2,0],
                'R': [1/2,0,1/2,0]}
    
    return np.array([encoding.get(base, [0,0,0,0]) for base in seq])


# load and encoder sequence data
def load_data():
    # fetch dataset
    raw_sequences = fetch_ucirepo(id=69)

    # extract DNA sequences and labels
    sequences = raw_sequences.data.features.values
    labels = raw_sequences.data.targets.values.reshape(-1,)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    # to torch tensor
    encoded_sequences = np.array([encode_dna(seq) for seq in sequences])
    encoded_sequences = torch.FloatTensor(encoded_sequences).permute(0, 2, 1)
    labels = torch.LongTensor(y)

    return encoded_sequences, labels


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()

    for sequences, labels in train_loader:
        optimizer.zero_grad()

        # forward pass
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)

        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()


def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    classification_accuracy = correct / total

    return classification_accuracy


def shuffle_several_times(s):
    s = np.squeeze(s)
    return dinuc_shuffle(s, num_shufs=100)


def deeplift_analysis(model, encoded_sequences, device):
    model.eval()

    ######################################
    # TODO: implement DeepLIFT
    ######################################


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load and preprocess data
    encoded_sequences, labels = load_data()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, labels, 
                                                        test_size=0.1, random_state=11)

    # dataloaders
    train_dataset = SpliceDataset(X_train, y_train)
    test_dataset = SpliceDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # init model
    model = SpliceCNN(input_size=60, num_classes=3, kernel_size=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train model
    num_epochs = 50
    for epoch in range(num_epochs):
        train_model(model, train_loader, criterion, optimizer, device)

        accuracy = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}")

    # perform DeepLIFT analysis
    deeplift_analysis(model, encoded_sequences, device)


if __name__ == "__main__":
    main()
