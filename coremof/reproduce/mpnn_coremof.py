"""
Implement of MPNN using torch_geometric.
You can also use dgllife: https://lifesci.dgl.ai/api/model.gnn.html?highlight=mpnn#module-dgllife.model.gnn.mpnn
https://lifesci.dgl.ai/api/model.zoo.html?highlight=mpnn#module-dgllife.model.model_zoo.mpnn_predictor
"""

from time import time
import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
from torch.nn.functional import mse_loss
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNN, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        prop = self.propagate(edge_index, x=x, norm=norm)
        return global_mean_pool(prop, batch)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


def main():
    x, y = torch.load('../data/coremof.pt')
    targets = ['LCD', 'PLD', 'D', 'ASA', 'NASA', 'AV']
    scales = [[y[:, i].mean().item(), y[:, i].std().item()] for i in range(y.shape[-1])]
    for i in range(y.shape[-1]):
        y[:, i] = (y[:, i] - scales[i][0]) / scales[i][1]
    pos = [i[:, :3] for i in x]
    test_size = int(0.1 * len(x))
    train_size = len(x) - test_size * 2
    graph_list = []
    for j in range(len(x)):
        dist = torch.cdist(pos[j], pos[j]) < 5
        edge_index = dense_to_sparse(dist)[0]
        graph = Data(x=x[j][:, 3].unsqueeze(-1), edge_index=edge_index, y=y[j])
        graph_list.append(graph)
    train_list, val_list, test_list = random_split(graph_list, [train_size, test_size, test_size])
    train_loader = DataLoader(train_list, batch_size=64)
    val_loader = DataLoader(val_list, batch_size=64)
    criterion = torch.nn.MSELoss()
    best_mse = [1e9] * 6

    for i in range(6):
        print(f'{"=" * 20} Start training {targets[i]} {"=" * 20}')
        model = MPNN(1, 1)
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
                pred = model(data.x.float(), data.edge_index, data.batch)
                loss = criterion(pred[..., 0], data.label.view(-1, y.shape[-1])[:, i])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs / train_size
            lr_scheduler.step()

            model.eval()
            mse = 0
            for data in val_loader:
                with torch.no_grad():
                    pred = model(data.x.float(), data.edge_index, data.batch)
                mse += mse_loss(pred[..., 0], data.label.view(-1, y.shape[-1])[:, i], reduction='sum').item()
            mse *= scales[i][1] / test_size
            best_mse[i] = min(best_mse[i], mse)
            print(f'Epoch: {epoch + 1:02d}, Loss: {total_loss:.4f}, MSE: {round(mse, 3)}, time: {time() - t0:.1f}s')
        print(f'Final MSE: {[round(x, 3) for x in best_mse[:i + 1]]}')


if __name__ == '__main__':
    main()
