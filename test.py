import torch
import numpy as np

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(3, 100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, 100)
        self.l4 = torch.nn.Linear(100, 100)
        self.l5 = torch.nn.Linear(100, 100)
        self.l7 = torch.nn.Linear(100, 100)
        self.l8 = torch.nn.Linear(100, 100)
        self.l6 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        x = torch.relu(self.l5(x))
        return torch.relu(self.l6(x))


def prediction(params_path, data):
    model = Net()
    model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))
    model.eval()

    data = np.asarray(data)
    data = np.ascontiguousarray(data.T)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    data = data.reshape(1, -1, data.shape[-2], data.shape[-1])

    with torch.no_grad():
        pred = model(data)
    pred = pred.detach().numpy().squeeze()
    pred = np.ascontiguousarray(pred.T)

    return pred