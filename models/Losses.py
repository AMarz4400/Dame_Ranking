import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_output, neg_output):
        # pos_output: scores for positive items
        # neg_output: scores for negative items
        loss = -torch.mean(torch.log(self.sigmoid(pos_output - neg_output) + 1e-8))
        return loss
