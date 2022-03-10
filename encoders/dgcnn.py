import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import itertools


class Flatten(nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out  # (batch_size, *size)


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = (
            torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)
            * num_points
        )
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)  # (batch_size, 3, num_points)

    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, 3)
    x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, 3)

    x = torch.matmul(
        x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(2)
    )  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, 9).transpose(2, 1)  # (batch_size, 9, num_points)

    x = torch.cat((pts, x), dim=1)  # (batch_size, 12, num_points)

    return x


def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    x = x.view(batch_size * num_points, -1)[
        idx, :
    ]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(
        batch_size, num_points, -1, num_dims
    )  # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)  # (batch_size, num_points, num_dims)

    return x


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[
        idx, :
    ]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(
        batch_size, num_points, k, num_dims
    )  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(
        1, 1, k, 1
    )  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature - x, x), dim=3).permute(
        0, 3, 1, 2
    )  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2


# Clustering layer definition (see DCEC article for equations)
class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        # represent weight as module parameter
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        #         print(x.shape)
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, alpha={}".format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


class DGCNNEncoder(nn.Module):
    def __init__(self, num_features, k=20):
        super(DGCNNEncoder, self).__init__()
        self.k = k
        self.num_features = num_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.clustering = None
        self.lin_features_len = 512
        if self.num_features < self.lin_features_len:
            self.flatten = Flatten()
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

    def forward(self, x):
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x0 = self.conv5(x)
        x = x0.max(dim=-1, keepdim=False)[0]
        feat = x.unsqueeze(1)

        if (
            self.num_features < self.lin_features_len
            or self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features


class FoldNetEncoder(nn.Module):
    def __init__(self, num_features, k):
        super(FoldNetEncoder, self).__init__()
        if k is None:
            self.k = 16
        else:
            self.k = k
        self.n = 2048
        self.num_features = num_features
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
        )
        self.clustering = None
        self.lin_features_len = 512
        if self.num_features < self.lin_features_len:
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)
        batch_size = pts.size(0)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)
        x = self.mlp1(x)
        x = self.graph_layer(x, idx)
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.mlp2(x)
        feat = x.transpose(2, 1)
        if (
            self.num_features < self.lin_features_len
            or self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features
