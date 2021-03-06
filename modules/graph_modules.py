import torch
from torch import nn


class GraphFilter(nn.Module):
    def __init__(self, grid_dims, graph_r, graph_eps, graph_lam):
        super(GraphFilter, self).__init__()
        self.grid_dims = grid_dims
        self.graph_r = graph_r
        self.graph_eps_sqr = graph_eps * graph_eps
        self.graph_lam = graph_lam

    def forward(self, grid, pc):
        # Data preparation
        bs_cur = pc.shape[0]
        grid_exp = grid.contiguous().view(
            bs_cur, self.grid_dims[0], self.grid_dims[1], 2
        )  # batch_size X dim0 X dim1 X 2
        pc_exp = pc.contiguous().view(
            bs_cur, self.grid_dims[0], self.grid_dims[1], 3
        )  # batch_size X dim0 X dim1 X 3
        graph_feature = torch.cat((grid_exp, pc_exp), dim=3).permute([0, 3, 1, 2])

        # Compute the graph weights
        wght_hori = (
            graph_feature[:, :, :-1, :] - graph_feature[:, :, 1:, :]
        )  # horizontal weights
        wght_vert = (
            graph_feature[:, :, :, :-1] - graph_feature[:, :, :, 1:]
        )  # vertical weights
        wght_hori = torch.exp(
            -torch.sum(wght_hori * wght_hori, dim=1) / self.graph_eps_sqr
        )  # Gaussian weight
        wght_vert = torch.exp(
            -torch.sum(wght_vert * wght_vert, dim=1) / self.graph_eps_sqr
        )
        wght_hori = (wght_hori > self.graph_r) * wght_hori
        wght_vert = (wght_vert > self.graph_r) * wght_vert
        wght_lft = torch.cat(
            (torch.zeros([bs_cur, 1, self.grid_dims[1]]).cuda(), wght_hori), 1
        )  # add left
        wght_rgh = torch.cat(
            (wght_hori, torch.zeros([bs_cur, 1, self.grid_dims[1]]).cuda()), 1
        )  # add right
        wght_top = torch.cat(
            (torch.zeros([bs_cur, self.grid_dims[0], 1]).cuda(), wght_vert), 2
        )  # add top
        wght_bot = torch.cat(
            (wght_vert, torch.zeros([bs_cur, self.grid_dims[0], 1]).cuda()), 2
        )  # add bottom
        wght_all = torch.cat(
            (
                wght_lft.unsqueeze(1),
                wght_rgh.unsqueeze(1),
                wght_top.unsqueeze(1),
                wght_bot.unsqueeze(1),
            ),
            1,
        )

        # Perform the actural graph filtering: x = (I - \lambda L) * x
        wght_hori = wght_hori.unsqueeze(1).expand(-1, 3, -1, -1)  # dimension expansion
        wght_vert = wght_vert.unsqueeze(1).expand(-1, 3, -1, -1)
        pc = (
            pc.permute([0, 2, 1])
            .contiguous()
            .view(bs_cur, 3, self.grid_dims[0], self.grid_dims[1])
        )
        pc_filt = (
            torch.cat(
                (
                    torch.zeros([bs_cur, 3, 1, self.grid_dims[1]]).cuda(),
                    pc[:, :, :-1, :] * wght_hori,
                ),
                2,
            )
            + torch.cat(
                (
                    pc[:, :, 1:, :] * wght_hori,
                    torch.zeros([bs_cur, 3, 1, self.grid_dims[1]]).cuda(),
                ),
                2,
            )
            + torch.cat(
                (
                    torch.zeros([bs_cur, 3, self.grid_dims[0], 1]).cuda(),
                    pc[:, :, :, :-1] * wght_vert,
                ),
                3,
            )
            + torch.cat(
                (
                    pc[:, :, :, 1:] * wght_vert,
                    torch.zeros([bs_cur, 3, self.grid_dims[0], 1]).cuda(),
                ),
                3,
            )
        )  # left, right, top, bottom

        pc_filt = pc + self.graph_lam * (
            pc_filt - torch.sum(wght_all, dim=1).unsqueeze(1).expand(-1, 3, -1, -1) * pc
        )  # equivalent to ( I - \lambda L) * x
        pc_filt = pc_filt.view(bs_cur, 3, -1).permute([0, 2, 1])
        return pc_filt, wght_all
