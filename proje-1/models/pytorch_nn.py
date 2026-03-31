import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PyTorchModelArchitecture(nn.Module):
    def __init__(self, layer_dims, random_seed=42):
        super(PyTorchModelArchitecture, self).__init__()
        torch.manual_seed(random_seed)
        
        layers = []
        for i in range(1, len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.network = nn.Sequential(*layers)
        
        # Ağırlıkların He Initialization ile başlatılması
        for i in range(1, len(layer_dims)):
            std_val = float(np.sqrt(2. / layer_dims[i-1]))
            nn.init.normal_(self.network[2*(i-1)].weight, mean=0.0, std=std_val)
            nn.init.zeros_(self.network[2*(i-1)].bias)
                
    def forward(self, x):
        return self.network(x)

class PyTorchTrainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
    def fit(self, X_train_np, y_train_np, epochs=1000, verbose=True):
        print("\nPyTorch eğitimi başlıyor...")
        X = torch.tensor(X_train_np, dtype=torch.float32)
        Y = torch.tensor(y_train_np.flatten(), dtype=torch.long)
        
        history = {'train_loss': []}
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, Y)
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                history['train_loss'].append(loss.item())
                if verbose:
                    print(f"Epoch {epoch:5d} -> Eğitim Kaybı (Train Loss) = {loss.item():.4f}")
                    
        return history
        
    def predict(self, X_test_np):
        X = torch.tensor(X_test_np, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy().reshape(-1, 1)
