import numpy as np
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def load_data(path):
    data = pd.read_csv(path).iloc[:, 1:]
    keys, dataset = data.keys().to_list(), data.values
    return keys, np.asarray(dataset)


def transfrom_data(original_dataset, window_size = 7):
    x, y = [], []
    for i in range(original_dataset.shape[1]):
        #stder = StandardScaler()
        #original_dataset[:, i] = stder.fit_transform(original_dataset[:, i].reshape(-1, 1)).reshape(-1)
        pass
    for i in range(original_dataset.shape[0]-window_size):
        x.append(original_dataset[i:i+window_size])
        y.append(original_dataset[i+window_size])
    return np.asarray(x), np.asarray(y)


class TimeSerise(torch.nn.Module):
    
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.len_seq, self.n_feature = input_size
        self.output_size = output_size
        self.layer_1 = torch.nn.Parameter(torch.randn((self.n_feature, 4)), requires_grad=True)
        self.relu = torch.nn.LeakyReLU()
        self.layer_2 = torch.nn.Parameter(torch.randn((4, self.output_size)), requires_grad=True)

    def forward(self, X):
        outputs = torch.matmul(X, self.layer_1)
        outputs = self.relu(outputs)
        outputs = torch.matmul(outputs, self.layer_2)
        outputs = torch.sum(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    
    data_path = './data/DailyDelhiClimateTrain.csv'
    keys, dataset = load_data(data_path)
    x, y = transfrom_data(dataset.copy())
    x, y = torch.as_tensor(x, dtype=torch.float), torch.as_tensor(y, dtype=torch.float)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    print(f"x_shape = {x.shape}, y_shape = {y.shape}")

    model = TimeSerise(x.shape[1:], y.shape[1])
    cri = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)
    epochs = 5000

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        y_hat = model(x_train)
        loss = cri(y_train, y_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            with torch.no_grad():
                y_hat = model(x_val)
                val_loss = cri(y_val, y_hat)
                train_losses.append(loss.detach().numpy())
                val_losses.append(val_loss.detach().numpy())

                print(f'epoch {epoch}th -> train_loss = {loss/x_train.shape[0]} and val_loss = {val_loss/x_val.shape[0]}')
                for i in range(len(keys)):
                    n_samples = 100
                    plt.subplot(2, 2, i+1)
                    plt.plot(y_val[:n_samples, i], color='blue')
                    plt.plot(y_hat[:n_samples, i], color='orange')
                    plt.title(keys[i])

                plt.show(block=False)
                plt.pause(0.5)
                plt.close()    

    #plt.plot(train_losses[10:], color='blue')
    #plt.plot(val_losses[10:], color='orange')
    #plt.show()

    with torch.no_grad():
        data_path = './data/DailyDelhiClimateTest.csv'
        _, test_dataset = load_data(data_path)
        x_test, y_test = transfrom_data(test_dataset.copy())
        x_test, y_test = torch.as_tensor(x_test, dtype=torch.float), torch.as_tensor(y_test, dtype=torch.float)
        y_hat = model(x_test)
        test_loss = cri(y_test, y_hat)
        print(f'test_loss = {test_loss/x_test.shape[0]}')

        for i in range(len(keys)):
            n_samples = 100
            plt.subplot(2, 2, i+1)
            plt.plot(y_test[:n_samples, i], color='blue')
            plt.plot(y_hat[:n_samples, i], color='orange')
            plt.title(keys[i])
        plt.show()