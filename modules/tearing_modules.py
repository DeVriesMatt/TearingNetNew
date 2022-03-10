import torch
from torch import nn


class TearingNetBasic(nn.Module):
    def __init__(
        self,
        tearing1_dims=[517, 256, 128, 64],
        tearing2_dims=[581, 256, 128, 2],
        grid_dims=[45, 45],
        kernel_size=1,
    ):
        super(TearingNetBasic, self).__init__()

        self.grid_dims = grid_dims

        self.tearing1 = nn.Sequential(
            nn.Conv2d(
                tearing1_dims[0],
                tearing1_dims[1],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                tearing1_dims[1],
                tearing1_dims[2],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                tearing1_dims[2],
                tearing1_dims[3],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.tearing2 = nn.Sequential(
            nn.Conv2d(
                tearing2_dims[0],
                tearing2_dims[1],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                tearing2_dims[1],
                tearing2_dims[2],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                tearing2_dims[2],
                tearing2_dims[3],
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, cw, grid, pc, **kwargs):
        grid_exp = grid.contiguous().view(
            grid.shape[0], self.grid_dims[0], self.grid_dims[1], 2
        )  # batch_size X dim0 X dim1 X 2

        pc_exp = pc.contiguous().view(
            pc.shape[0], self.grid_dims[0], self.grid_dims[1], 3
        )  # batch_size X dim0 X dim1 X 3

        cw_exp = (
            cw.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self.grid_dims[0], self.grid_dims[1], -1)
        )
        # batch_size X dim0 X dim1 X code_length

        in1 = torch.cat((grid_exp, pc_exp, cw_exp), 3).permute([0, 3, 1, 2])

        # Compute the torn 2D grid
        out1 = self.tearing1(in1)  # 1st tearing
        in2 = torch.cat((in1, out1), 1)
        out2 = self.tearing2(in2)  # 2nd tearing
        out2 = (
            out2.permute([0, 2, 3, 1])
            .contiguous()
            .view(grid.shape[0], self.grid_dims[0] * self.grid_dims[1], 2)
        )
        return grid + out2
