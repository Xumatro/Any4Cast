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

    def forward(self, x, h):
        out, h = self.gru(x, h.detach())
        out = self.fc(out[:, -1, :])
        return out, h

    def train(self, dataset, quiet=False):
        losses = []

        [x_train, y_train, _, _] = [torch.tensor(
            ds, device=self.device) for ds in dataset.create_datasets()]

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lrn_rate)

        h = torch.zeros(self.n_layers, x_train.size(
            0), self.hid_dim, device=self.device).requires_grad_()

        for i in range(self.n_epochs):
            prediction, h = self(x_train, h)
            loss = self.criterion(prediction, y_train)
            losses.append(loss.item())

            if not quiet:
                print(f"Epoch: {i + 1},\tMSE: {loss.item()}")

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return losses
