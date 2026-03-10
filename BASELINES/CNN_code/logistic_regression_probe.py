import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def create_data_loader(train_x, corrupted_trainy, test_x, original_testy, batch_size=128):

    train_x = train_x.float()
    test_x = test_x.float()

    corrupted_trainy = corrupted_trainy.long()
    original_testy = original_testy.long()

    train_dataset = TensorDataset(train_x, corrupted_trainy)
    test_dataset = TensorDataset(test_x, original_testy)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)
    
    
def training_probe(model, train_loader, dev, epochs=20, lr=1e-3, batch_size=128):
    
    model = model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(dev), y_batch.to(dev)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)        
            loss.backward()                     
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

    
def inference(model, test_loader, dev):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(dev), y_batch.to(dev)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy
