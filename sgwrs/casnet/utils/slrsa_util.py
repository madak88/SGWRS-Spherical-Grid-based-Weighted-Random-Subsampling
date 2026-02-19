import torch
import torch.nn as nn
import torch.nn.functional as F

from casnet.utils.util import better_query_ball_point, index_points

class RelationEncoding(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp):
        super(RelationEncoding, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, points):
        """
            relation encoding on original points, no sampling
            :param points: input original points, [B, C, N]
            :return: new points after relation encoding, [B, D, N]
        """
        points = points.permute(0, 2, 1)
        B, N, C = points.shape

        # grouping points
        idx = better_query_ball_point(self.radius, self.nsample, points, points)  # [B, N, nsample]
        grouped_points = index_points(points, idx)  # [B, N, nsample, C]

        # encoding
        points = points.view(B, N, 1, C).repeat(1, 1, self.nsample, 1)  # [B, N, nsample, C]
        edge_fea = torch.cat([points, grouped_points-points],  dim=-1)  # [B, N, nsample, 2C]
        edge_fea = edge_fea.permute(0, 3, 2, 1)  # [B, 2C, nsample, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            edge_fea = F.leaky_relu(bn(conv(edge_fea)), negative_slope=0.2)
        new_points = torch.max(edge_fea, 2)[0]  # [B, D, N]

        return new_points

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if self.group_all:
            sampling_channel = in_channel // 2
        else:
            sampling_channel = (in_channel - 6) // 2 + 3

        self.SLRS = SLRS(sampling_channel, npoint)


    def forward(self, xyz, points, gamma=1, hard=False):
        """
        set abstraction level
        :param xyz: input points position data, [B, C, N]
        :param points: input points data, [B, D, N]
        :param hard: whether to use straight through, Bool

        :return sample_matrix, cos_loss
        """

        xyz = xyz.permute(0, 2, 1)  # [B, N, C]
        if points is not None: points = points.permute(0, 2, 1)  # [B, N, D]
        else: points = xyz
        grouped_points_agg = torch.cat((points, xyz), dim=-1)
        select, Cos_loss, _ = self.SLRS(grouped_points_agg, xyz, hard=hard, gamma=gamma)  # [B, npoint, D], [B, npoint, N]

        return select, Cos_loss

class SLRS(nn.Module):

    def __init__(self, in_features, select_N,seg=False):
        super(SLRS, self).__init__()
        self.seg = seg
        self.w1 = nn.Conv2d(in_features, in_features, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.w2 = nn.Conv2d(in_features, select_N, 1, 1)
        self.bn2 = nn.BatchNorm2d(select_N)

    def forward(self, points, xyz, hard=False, gamma=1):

        """
        selecting module
        :param x: local region descriptor, [B, N, D]
        :param hard: whether to use straight through
        :return: selecting matrix ret, [B, select_N, N]
                 cosine loss
        """
        device = points.device
        B, N, D = points.shape

        points = points.permute(0, 2, 1).view(B, D, N, 1)
        points = self.bn1(self.w1(points))
        # print("points",points.shape)
        points = self.bn2(self.w2(points))

        select_weights = points.squeeze(-1)
        B, select_N, N = select_weights.shape
        if self.seg:
            # # normal distribution
            normals = torch.randn_like(select_weights)  # ~N(0,1)
            select_weights = (select_weights + 5*normals) / gamma
        else:
            select_weights = select_weights / gamma
        select_weights = select_weights.softmax(dim=-1)

        cos_loss = 0
        if select_N != 1:
            # cosine loss
            inner_product = select_weights.matmul(select_weights.permute(0, 2, 1))
            norm = torch.sqrt(select_weights.mul(select_weights).sum(dim=-1, keepdim=True))
            norm_matrix = norm.matmul(norm.permute(0, 2, 1))
            cosine_matrix = torch.div(inner_product, norm_matrix.add_(1e-10))
            ones = torch.ones([B, select_N, select_N], device=device)
            I = torch.eye(select_N, device=device).view(1, select_N, select_N).repeat(B, 1, 1)
            cosine_matrix_nodiag = cosine_matrix.mul(ones - I)
            cos_loss = torch.sqrt(torch.sum(cosine_matrix_nodiag.mul(cosine_matrix_nodiag), dim=(1, 2))).mean()

        if hard:
            # Straight through.
            index = select_weights.max(dim=-1, keepdim=True)[1]
            # print("index", index.shape)
            select_hard = torch.zeros_like(select_weights).scatter_(-1, index, 1.0)
            ret = select_hard - select_weights.detach() + select_weights
        else:
            ret = select_weights

        index = torch.max(ret, dim=-1)[1]
        unique = torch.unique(index[0, :], return_counts=True)[0]
        # return ret
        return ret, cos_loss, unique.shape
