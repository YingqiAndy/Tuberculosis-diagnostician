import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class Training:
    """
    wrapped class to train
    """
    def __init__(self, model,
                 train_loader,
                 test_loader=None,
                 criterion=nn.CrossEntropyLoss(),
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = None
        self.device = torch.device(device)
        self.model.to(self.device)

    def set_optimizer(self, optimizer=None, learning_rate=0.001):
        """
        Set optimizer
        :param optimizer: torch.optim class. Defaults to Adam.
        :param learning_rate: float. Learning rate. Defaults to 0.001
        """
        if optimizer is None:
            optimizer = optim.Adam
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs=20):
        """
        Train model. Print the loss and time for each epoch and the loss curve, and save the local optimized model
        """
        loss_list = []
        best_loss = float('inf')

        for epoch in range(num_epochs):

            start_time = time.time()
            train_loss = 0
            self.model.train()

            for images, labels in self.train_loader:

                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                loss.backward()
                train_loss += loss.item() * images.size(0)
                self.optimizer.step()

            train_loss = train_loss / len(self.train_loader.dataset)
            epoch_time = time.time() - start_time
            loss_list.append(train_loss)
            print('Epoch: {} \tTraining Loss: {:.6f} \tTime: {:.2f}s'.format(epoch+1, train_loss, epoch_time))

            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model("classifier.pth")

        plt.figure(figsize=(12, 9))
        plt.plot(range(num_epochs), loss_list)
        plt.xlabel('epochs')
        plt.ylabel('loss value: Cross Entropy Loss')
        plt.show()
        plt.savefig('loss.png')

    def evaluate(self):
        """
        Evaluate model. Prints accuracy, precision, recall, F1 score and confusion matrix.
        """
        self.model.eval()
        all_predicted = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predicted)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predicted, average='weighted')
        confusion_mat = confusion_matrix(all_labels, all_predicted)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        metrics = pd.DataFrame({"accuracy": [accuracy],
                                "precision": [precision],
                                "recall": [recall],
                                "f1": [f1]})
        metrics.to_csv("metrics.csv", index=False)

        sns.set()
        plt.figure(figsize=(12, 12))
        sns.heatmap(confusion_mat, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig("confusion_matrix.png")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
