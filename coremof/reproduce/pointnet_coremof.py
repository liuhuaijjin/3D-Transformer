"""
Implementation of PointNet++ using torch_geometric.
"""
from time import time
import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
from torch.nn.functional import mse_loss
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels), ReLU(), Linear(out_channels, out_channels))

    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.


class PointNet(torch.nn.Module):
    def __init__(self, tgt):
        super(PointNet, self).__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, tgt)

    def forward(self, pos, batch):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.
        edge_index = knn_graph(pos, k=3, batch=batch, loop=True)

        # 3. Start bipartite message passing.
        h = self.conv1(h=pos, pos=pos, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=pos, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # 5. Classifier.
        return self.classifier(h)


def main():
    x, y = torch.load('../data/coremof.pt')
    targets = ['LCD', 'PLD', 'D', 'ASA', 'NASA', 'AV']
    scales = [[y[:, i].mean().item(), y[:, i].std().item()] for i in range(y.shape[-1])]
    for i in range(y.shape[-1]):
        y[:, i] = (y[:, i] - scales[i][0]) / scales[i][1]
    atom = [i[:, 3] for i in x]
    pos = [i[:, :3] for i in x]
    graph_list = []
    for i in range(len(x)):
        graph = Data(x=atom[i], pos=pos[i], y=y[i])
        graph_list.append(graph)

    test_size = int(0.1 * len(x))
    train_size = len(x) - test_size * 2
    train_list, val_list, test_list = random_split(graph_list, [train_size, test_size, test_size])
    train_loader = DataLoader(train_list, batch_size=32)
    val_loader = DataLoader(val_list, batch_size=32)

    criterion = torch.nn.MSELoss()
    best_mse = [1e9] * 6

    for i in range(6):
        print(f'{"=" * 20} Start training {targets[i]} {"=" * 20}')
        model = PointNet(1)
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epo: 0.99 ** epo)

        for epoch in range(50):
            t0 = time()
            model.train()
            total_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                pred = model(data.pos.float(), data.batch)
                loss = criterion(pred[..., 0], data.label.view(-1, y.shape[-1])[:, i])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs / train_size
            lr_scheduler.step()

            model.eval()
            mse = 0
            for data in val_loader:
                with torch.no_grad():
                    pred = model(data.pos.float(), data.batch)
                mse += mse_loss(pred[..., 0], data.label.view(-1, y.shape[-1])[:, i], reduction='sum').item()
            mse *= scales[i][1] / test_size
            best_mse[i] = min(best_mse[i], mse)
            print(f'Epoch: {epoch + 1:02d}, Loss: {total_loss:.4f}, MSE: {round(mse, 3)}, time: {time() - t0:.1f}s')
        print(f'Final MSE: {[round(x, 3) for x in best_mse[:i + 1]]}')


if __name__ == '__main__':
    main()
























