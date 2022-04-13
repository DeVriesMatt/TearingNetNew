import torch
from torch import nn

from modules.folding_modules import FoldingModule


class FoldingNetBasicDecoder(nn.Module):
    def __init__(self, num_clusters, num_features):
        super(FoldingNetBasicDecoder, self).__init__()

        # initialise deembedding
        self.lin_features_len = 512
        self.num_features = num_features
        self.num_cluster = num_clusters
        if self.num_features < self.lin_features_len:
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

        # make grid
        range_x = torch.linspace(-3.0, 3.0, 45)
        range_y = torch.linspace(-3.0, 3.0, 45)
        x_coor, y_coor = torch.meshgrid(range_x, range_y)
        self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

        # initialise folding module
        self.folding = FoldingModule()

    def forward(self, x):
        if self.num_features < self.lin_features_len:
            x = self.deembedding(x)
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(1)

        grid = self.grid.cuda().unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs = self.folding(x, grid)
        return outputs, grid


if __name__ == "__main__":
    model = FoldingNetBasicDecoder(10, 50).cuda()
    inputs = torch.ones((1, 50)).cuda()
    pc0 = model(inputs)
