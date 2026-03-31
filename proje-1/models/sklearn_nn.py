from sklearn.neural_network import MLPClassifier

class SklearnModel:
    """
    Scikit-Learn kütüphanesinin sağladığı MLPClassifier yapısı kullanılarak oluşturulan temel model sınıfı.
    """
    def __init__(self, hidden_layer_sizes=(6,), max_iter=1000, learning_rate_init=0.01, random_seed=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='sgd',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_seed,
            batch_size=32
        )
        
    def fit(self, X_train, y_train):
        print("\nScikit-Learn modeli eğitimi başlıyor...")
        self.model.fit(X_train, y_train.flatten())
        return {'train_loss': self.model.loss_curve_}
        
    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return preds.reshape(-1, 1)
