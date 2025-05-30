import os
import torch 
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision.models import vgg16



class SVM:
    def __init__(self, model_path= "svm_model.pkl"):
        self.model_path = model_path 
        self.svm_model = None  

    def load_pretrained_model(self, device):
        vgg = vgg16(pretrained= True)
        model = torch.nn.Sequential(
            vgg.features,
            vgg.avgpool,
            torch.nn.Flatten()
        )
        return model.eval().to(device)
    
    def train(self, data_train):
        X_train, y_train = data_train
        self.svm_model = SVC(kernel="linear", probability=True)
        self.svm_model.fit(X_train, y_train)

        # save model 
        joblib.dump(self.svm_model, self.model_path)
        print(f"Saving SVM model to {self.model_path}")

    def evaluate(self, data_test): 
        X_test, y_test = data_test
        if self.svm_model is None: 
            if os.path.exists(self.model_path): 
                # load from trained model 
                self.svm_model = joblib.load(self.model_path) 
            else: 
                raise ValueError("No trained model found. Train it first.")
        
        print("Evaluating SMV......")
        y_pred = self.svm_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")