import torch
import torch.nn as nn




class Model(nn.Module):
    def __init__(self, ch_in=3, n=16):
        super(Model, self).__init__()
        self.conv_gv = nn.Sequential(
            nn.Conv1d(ch_in, n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(n, n, 5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.MaxPool1d(5, stride=2, return_indices=False),

            nn.Conv1d(n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(5, stride=2, return_indices=False),

            nn.Conv1d(2*n, 4*n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(4*n, 4*n, 5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.MaxPool1d(5, stride=2, return_indices=False),

            nn.Conv1d(4*n, 8*n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(8*n, 8*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(5, stride=2, return_indices=False),

            nn.Conv1d(8*n, 16*n, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16*n, 16*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(5, stride=2, return_indices=False))

        self.conv_lv = nn.Sequential(
            nn.Conv1d(ch_in, n, 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.Conv1d(n, n, 5, stride=1, padding=2),
            #nn.ReLU(),
            nn.Conv1d(n, n, 5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.MaxPool1d(7, stride=2, return_indices=False),

            nn.Conv1d(n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            #nn.ReLU(),
            nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(7, stride=2, return_indices=False))

        self.conv_sv = nn.Sequential(
            nn.Conv1d(ch_in, n, 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.Conv1d(n, n, 5, stride=1, padding=2),
            #nn.ReLU(),
            nn.Conv1d(n, n, 5, stride=1, padding=2),
            nn.ReLU(),
            
            nn.MaxPool1d(7, stride=2, return_indices=False),

            nn.Conv1d(n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),
            #nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            #nn.ReLU(),
            nn.Conv1d(2*n, 2*n, 5, stride=1, padding=2),
            nn.ReLU(),

            nn.MaxPool1d(7, stride=2, return_indices=False))

        self.linear = nn.Sequential(
            nn.Linear(92*n, 64*n),
            nn.ReLU(),
            #nn.Linear(32*n, 32*n),
            #nn.ReLU(),
            #nn.Linear(32*n, 32*n),
            #nn.ReLU(),
            #nn.Linear(32*n, 32*n),
            #nn.ReLU(),
            nn.Linear(64*n, 1),
            nn.Sigmoid())

        self.drop = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, gv, lv, sv):
        gv_out = self.conv_gv(gv)
        lv_out = self.conv_lv(lv)
        sv_out = self.conv_lv(sv)
        gv_out = gv_out.view(gv_out.shape[0], -1)
        lv_out = lv_out.view(lv_out.shape[0], -1)
        sv_out = sv_out.view(sv_out.shape[0], -1)
        #print(gv_out.size(), lv_out.size(), psd.size())
        out = torch.cat((gv_out, lv_out, sv_out), dim=1)
        #out = self.drop(out)
        #print(out.size())
        out = self.linear(out)

        return out
