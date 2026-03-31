import numpy as np

class CustomMLP:
    """
    Çoklu Sınıflandırma (Softmax & Cross-Entropy) yapısına sahip Çok Katmanlı Algılayıcı (MLP).
    """
    def __init__(self, layer_dims, random_seed=42):
        self.layer_dims = layer_dims
        self.parameters = {}
        self.L = len(layer_dims) - 1
        
        np.random.seed(random_seed)
        
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
            
    def _relu(self, Z):
        return np.maximum(0, Z)
        
    def _relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
        
    def _softmax(self, Z):
        # Nümerik stabilite için max değer çıkarıldı
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)
        
    def _forward_propagation(self, X):
        caches = {}
        A = X
        
        for l in range(1, self.L):
            Z = np.dot(self.parameters['W' + str(l)], A) + self.parameters['b' + str(l)]
            A = self._relu(Z)
            caches['Z' + str(l)] = Z
            caches['A' + str(l)] = A
            
        Z = np.dot(self.parameters['W' + str(self.L)], A) + self.parameters['b' + str(self.L)]
        AL = self._softmax(Z)
        caches['Z' + str(self.L)] = Z
        caches['A' + str(self.L)] = AL
        
        return AL, caches
        
    def _compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        return cost
        
    def _backward_propagation(self, AL, Y, caches, X):
        grads = {}
        m = Y.shape[1]
        
        dZ = AL - Y
        grads['dW' + str(self.L)] = np.dot(dZ, caches['A' + str(self.L - 1)].T) / m
        grads['db' + str(self.L)] = np.sum(dZ, axis=1, keepdims=True) / m
        
        dA_prev = np.dot(self.parameters['W' + str(self.L)].T, dZ)
        
        for l in reversed(range(1, self.L)):
            dZ = self._relu_backward(dA_prev, caches['Z' + str(l)])
            A_prev = caches['A' + str(l-1)] if l > 1 else X
            grads['dW' + str(l)] = np.dot(dZ, A_prev.T) / m
            grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if l > 1:
                dA_prev = np.dot(self.parameters['W' + str(l)].T, dZ)
                
        return grads
        
    def _update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
            
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, learning_rate=0.01, verbose=True):
        m = X_train.shape[0]
        num_classes = self.layer_dims[-1]
        
        X = X_train.T
        
        Y_one_hot = np.zeros((num_classes, m))
        Y_one_hot[y_train.flatten(), np.arange(m)] = 1
        
        if X_val is not None:
            X_v = X_val.T
            m_v = X_val.shape[0]
            Y_val_one_hot = np.zeros((num_classes, m_v))
            Y_val_one_hot[y_val.flatten(), np.arange(m_v)] = 1
        
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"NumPy Modeli Eğitimi Başlıyor ({epochs} Epochs)...")
        for i in range(epochs):
            AL, caches = self._forward_propagation(X)
            cost = self._compute_cost(AL, Y_one_hot)
            
            grads = self._backward_propagation(AL, Y_one_hot, caches, X)
            self._update_parameters(grads, learning_rate)
            
            if i % 10 == 0 or i == epochs - 1:
                history['train_loss'].append(cost)
                if X_val is not None:
                    AL_v, _ = self._forward_propagation(X_v)
                    val_cost = self._compute_cost(AL_v, Y_val_one_hot)
                    history['val_loss'].append(val_cost)
                    if verbose:
                        print(f"Epoch {i:5d} -> Eğitim Kaybı (Train Loss) = {cost:.4f} | Val Kaybı = {val_cost:.4f}")
                else:
                    if verbose:
                        print(f"Epoch {i:5d} -> Eğitim Kaybı (Train Loss) = {cost:.4f}")
                        
        return history

    def predict(self, X_test):
        X = X_test.T
        AL, _ = self._forward_propagation(X)
        predictions = np.argmax(AL, axis=0)
        return predictions.reshape(-1, 1)
