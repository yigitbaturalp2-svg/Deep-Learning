import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from models.base_nn import CustomMLP
from models.sklearn_nn import SklearnModel
from models.pytorch_nn import PyTorchModelArchitecture, PyTorchTrainer
from evaluator import Evaluator

def main():
    processor = DataProcessor("Depression Student Dataset.csv", random_seed=42)
    X_train, y_train, X_dev, y_dev, X_test, y_test = processor.process_data()
    
    input_size = processor.input_size
    num_classes = processor.num_classes
    labels = processor.label_encoder.classes_
    
    epochs = 1000
    learning_rate = 0.05   
    
    print("\n[MODÜL 1] - 2 Katmanlı NumPy Modeli")
    base_dims = [input_size, 10, num_classes] 
    custom_model = CustomMLP(layer_dims=base_dims, random_seed=42)
    custom_history = custom_model.fit(X_train, y_train, X_dev, y_dev, epochs=epochs, learning_rate=learning_rate, verbose=False)
    
    custom_preds = custom_model.predict(X_test)
    Evaluator.plot_learning_curves(custom_history, title="NumPy Temel Model (1 Gizli Katman) Öğrenme Eğrisi")
    Evaluator.evaluate_model(y_test, custom_preds, labels=labels, title="NumPy Temel Model Sonuçları")
    
    print("\n[MODÜL 2] - Çok Katmanlı NumPy Modeli")
    opt_dims = [input_size, 16, 8, num_classes]
    opt_model = CustomMLP(layer_dims=opt_dims, random_seed=42)
    opt_history = opt_model.fit(X_train, y_train, X_dev, y_dev, epochs=epochs, learning_rate=learning_rate, verbose=False)
    
    opt_preds = opt_model.predict(X_test)
    Evaluator.plot_learning_curves(opt_history, title="NumPy Çok Katmanlı Model Öğrenme Eğrisi")
    Evaluator.evaluate_model(y_test, opt_preds, labels=labels, title="NumPy Çok Katmanlı Model Sonuçları")
    
    print("\n[MODÜL 3] - Scikit-Learn Modeli")
    sk_model = SklearnModel(hidden_layer_sizes=(16, 8), max_iter=epochs, learning_rate_init=learning_rate, random_seed=42)
    sk_history = sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    
    Evaluator.plot_learning_curves(sk_history, title="Scikit-Learn Model Öğrenme Eğrisi")
    Evaluator.evaluate_model(y_test, sk_preds, labels=labels, title="Scikit-Learn Model Sonuçları")
    
    print("\n[MODÜL 4] - PyTorch Modeli")
    pt_arch = PyTorchModelArchitecture(layer_dims=opt_dims, random_seed=42)
    pt_trainer = PyTorchTrainer(model=pt_arch, learning_rate=learning_rate)
    pt_history = pt_trainer.fit(X_train, y_train, epochs=epochs, verbose=False)
    pt_preds = pt_trainer.predict(X_test)
    
    Evaluator.plot_learning_curves(pt_history, title="PyTorch Model Öğrenme Eğrisi")
    Evaluator.evaluate_model(y_test, pt_preds, labels=labels, title="PyTorch Model Sonuçları")

if __name__ == "__main__":
    main()
