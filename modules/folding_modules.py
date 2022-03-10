import torch
from torch import nn


class FoldingModule(nn.Module):
    def __init__(self):
        super(FoldingModule, self).__init__()

        self.folding1 = nn.Sequential(
            nn.Linear(512 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.folding2 = nn.Sequential(
            nn.Linear(512 + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x, grid):
        cw_exp = x.expand(-1, grid.shape[1], -1)

        cat1 = torch.cat((cw_exp, grid), dim=2)
        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((cw_exp, folding_result1), dim=2)
        folding_result2 = self.folding2(cat2)
        return folding_result2
