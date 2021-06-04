import torch.nn as nn
import torch

class Sknet(nn.Module):
    def __init__(self, M, G, features, r, stride=1, L=32):
        super(Sknet, self).__init__()
        d = max(int(features / r), L)
        self.M = M

        self.convs = nn.ModuleList([])
        for i in range(self.M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3 + 2 * i,
                        stride=stride,
                        padding= 1 + i,
                        groups=G
                    ), nn.BatchNorm2d(features), nn.ReLU(inplace=True)
                )
            )
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for j in range(self.M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)   # 增加维度
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)

        fea_u = torch.sum(feas, dim=1)
        fea_s = fea_u.mean(-1).mean(-1)       # 转换成z * 1 * 1
        fea_f = self.fc(fea_s)               # 转化成为 c * 1 * 1

        for j, fc in enumerate(self.fcs):
            print(j, fea_f.shape)
            vector = fc(fea_f).unsequeeze_(dim=1)
            if j == 0:
                attention_vector = vector
            else:
                attention_vector = torch.cat([attention_vector, vector], dim=1)

        attention_vector = self.softmax(attention_vector)        # 将得到的每一个注意力向量都激活
        attention_vectors = attention_vector.unsequeeze(-1).unsequeeze(-1)
        fea_attention = (attention_vectors * fea_f).sum(dim=1)

        return fea_attention





