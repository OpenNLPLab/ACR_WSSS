
import os
import sys
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D

## new
class TreeEnergyLoss(nn.Module):
    def __init__(self, configer=None):
        super(TreeEnergyLoss, self).__init__()

        self.weight = 0.4
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02)

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            high_feats = F.interpolate(high_feats, size=(h, w), mode='bilinear', align_corners=False)
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            N = unlabeled_ROIs.sum()

        prob = torch.softmax(preds, dim=1)

        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N

        return self.weight * tree_loss