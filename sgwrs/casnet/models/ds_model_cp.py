
import torch
import torch.nn as nn

from casnet.utils.slrsa_util import  RelationEncoding, SetAbstraction
from casnet.settings.transformer_module import OA

class CascadeAttentionBasedSampling(nn.Module):

    def __init__(self,ds_num =512,hard=False):
        super(CascadeAttentionBasedSampling, self).__init__()
        self.hard =hard
        self.ds_num =ds_num
        self.re = RelationEncoding(radius=2, nsample=32, in_channel=6, mlp=[64, 64])
        self.oa1 = OA(channels=64)
        self.oa2 = OA(channels=64)
        self.oa3 = OA(channels=64)
        self.sa1 = SetAbstraction(npoint=ds_num, radius=2, nsample=32, in_channel=192*2+6, mlp=[256, 256], group_all=False)

    def forward(self, xyz, gamma=1, hard=False):
        """
            classification task
            :param xyz: input points, [B, C ,N]
            :param hard: whether to use straight-through, Bool
            :return: sample_matrix, cos_loss
        """

        # Relation Encoding Layer
        points = self.re(xyz)

        fea1 = self.oa1(points)
        fea2 = self.oa2(fea1)
        fea3 = self.oa3(fea2)
        fea =torch.cat([fea1,fea2,fea3],dim=-2)

        # Set Abstraction Levels
        sample_matrix, cos_loss = self.sa1(xyz, fea, gamma=gamma, hard=self.hard)

        return sample_matrix, cos_loss
