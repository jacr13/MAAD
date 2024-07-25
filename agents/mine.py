import torch
import torch.nn as nn


class Mine(nn.Module):
    def __init__(self, input_size, hps, rms_obs):
        super(Mine, self).__init__()
        self.hps = hps
        self.mine_version = self.hps.mine_version

        self.layers = nn.Sequential(
            nn.Linear(input_size, self.hps.mine_hidsize),
            nn.ReLU(),
            nn.Linear(self.hps.mine_hidsize, self.hps.mine_hidsize),
            nn.ReLU(),
            nn.Linear(self.hps.mine_hidsize, 1),
        )

    def forward(self, x, y):
        batch_size = x.size(0)

        # shuffle y to be independant of x
        idx = torch.randperm(batch_size)
        y_shuffle = y[idx]

        # predictions
        pred_xy = self.layers(torch.cat([x, y], dim=1))
        pred_x_y = self.layers(torch.cat([x, y_shuffle], dim=1))

        mi = self.mi(pred_xy, pred_x_y, self.mine_version)
        return mi

    def mi(self, pred_xy, pred_x_y, version="MINE"):
        # versions MINE, MINEf, DRT (density ratio trick)
        if version == "MINE":
            mi = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        elif version == "MINEf":
            mi = torch.mean(pred_xy) - torch.mean(torch.exp(pred_x_y - 1))
        elif version == "DRT":
            raise NotImplementedError("version not implemented")
        else:
            raise NotImplementedError("version not implemented")
        return mi
