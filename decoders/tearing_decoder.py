import torch
from torch import nn

from modules.folding_modules import FoldingModule
from modules.graph_modules import GraphFilter
from modules.tearing_modules import TearingNetBasic


class TearingNetDecoder(nn.Module):
    def __init__(self, num_clusters, num_features):
        super(TearingNetDecoder, self).__init__()

        # initialise deembedding
        self.lin_features_len = 512
        self.num_features = num_features
        self.num_cluster = num_clusters
        if self.num_features < self.lin_features_len:
            self.embedding = nn.Linear(self.lin_features_len, num_clusters, bias=False)
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

        # Initialize the regular 2D grid
        range_x = torch.linspace(-3.0, 3.0, 45)
        range_y = torch.linspace(-3.0, 3.0, 45)
        x_coor, y_coor = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

        # initialise folding and tearing
        self.folding = FoldingModule()
        self.tearing = TearingNetBasic()
        self.graph_filter = GraphFilter([45, 45], 1e-12, 0.02, 0.5)

    def forward(self, x):

        if self.num_features < self.lin_features_len:
            x = self.deembedding(x)
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(1)


        grid0 = (
            self.grid.cuda().unsqueeze(0).expand(x.shape[0], -1, -1)
        )  # batch_size X point_num X 2
        pc0 = self.folding(x, grid0)  # Folding Network
        grid1 = self.tearing(x.squeeze(1), grid0, pc0)  # Tearing Network
        pc1 = self.folding(x, grid1)  # Folding Network
        pc2, graph_wght = self.graph_filter(grid1, pc1)  # Graph Filtering
        return pc0, pc1, pc2, grid1, graph_wght


if __name__ == "__main__":
    model = TearingNetDecoder(10, 50).cuda()
    inputs = torch.ones((1, 50)).cuda()
    pc0 = model(inputs)
