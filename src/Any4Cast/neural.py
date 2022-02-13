import torch


class Network(torch.nn.Module):
    def __init__(self, settings):
        super(Network, self).__init__()

        self.inp_dim = settings['n_input_dimensions']
        self.hid_dim = settings['n_hidden_dimensions']
        self.out_dim = settings['n_output_dimensions']
        self.n_layers = settings['n_layers']
        self.lrn_rate = settings['learning_rate']
        self.n_epochs = settings['n_epochs']

        self.gru = torch.nn.GRU(self.inp_dim, self.hid_dim,
                                self.n_layers, batch_first=True)
        self.fc = torch.nn.Linear(self.hid_dim, self.out_dim)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hid_dim,
                         device=self.device).requires_grad_()
        out, hn = self.gru(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out

    def fit(self, dataset, quiet=False):
        self.train()
        losses = []

        [x_train, y_train, _, _] = [torch.tensor(
            ds, device=self.device) for ds in dataset.create_datasets()]

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lrn_rate)

        for i in range(self.n_epochs):
            prediction = self(x_train)
            loss = self.criterion(prediction, y_train)
            losses.append(loss.item())

            if not quiet:
                print(f"Epoch: {i + 1},\tMSE: {loss.item()}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses

    def test(self, dataset):
        self.eval()
        [_, _, x_test, y_test] = [torch.tensor(
            ds, device=self.device) for ds in dataset.create_datasets()]

        predciction = self(x_test)
        return predciction, self.criterion(predciction, y_test).item()

    def predict(self, dataset):
        self.eval()
        history = dataset.create_datasets(prediction_mode=True)
        prediction = torch.Tensor.cpu(
            self(torch.tensor(history, device=self.device)).detach()).numpy()

        return dataset.denormalize(history), dataset.denormalize(prediction)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
