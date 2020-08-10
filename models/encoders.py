import torch
import torch.nn as nn
import math

class GCN_layer(nn.Module):
    """
    BasicBlock of object branch encoder.
    """
    def __init__(self, N, d_model):
        super(GCN_layer, self).__init__()
        self.N = N
        self.d_model = d_model

        self.linear = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.batchnorm = nn.BatchNorm1d(num_features=N)
        self.relu = nn.ReLU(inplace=True)

        self.init_paramenters()

    def init_paramenters(self):
        stdv = 1. / math.sqrt(self.d_model)
        self.linear.weight.data.uniform_(-stdv, stdv)

    def forward(self, relation_matrix, H_l):
        identity = H_l

        out = torch.bmm(relation_matrix, H_l)
        out = self.linear(out)
        out = self.batchnorm(out)
        out = identity + out
        ret = self.relu(out)

        return ret

class GCN(nn.Module):
    """
    Encoder of object branch.
    """
    def __init__(self, in_feature_size, out_feature_size, N, drop_probability=0.5):
        super(GCN, self).__init__()
        self.in_feature_size = in_feature_size
        self.out_feature_size = out_feature_size
        self.N = N
        self.d_model = out_feature_size
        self.drop_probability = drop_probability

        self.linear_weight0 = nn.Linear(in_features=in_feature_size, out_features=out_feature_size, bias=False)
        self.gcn_layer0 = GCN_layer(N=self.N, d_model=self.d_model)
        self.drop0 = nn.Dropout(p=self.drop_probability, inplace=False)
        self.gcn_layer1 = GCN_layer(N=self.N, d_model=self.d_model)
        self.drop1 = nn.Dropout(p=self.drop_probability, inplace=False)
        self.gcn_layer2 = GCN_layer(N=self.N, d_model=self.d_model)
        self.avg_pool = nn.AvgPool1d(kernel_size=5, stride=5)

        self.init_paramenters()

    def init_paramenters(self):
        stdv = 1. / math.sqrt(self.out_feature_size)
        self.linear_weight0.weight.data.uniform_(-stdv, stdv)

    def forward(self, G_st, F_0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        diag = torch.sum(G_st, dim=-1, keepdim=True)
        diag = torch.eye(self.N).to(torch.float32)[None, :].to(device) * diag
        diag = 1. / torch.sqrt(diag)

        relation_matrix = torch.bmm(diag, G_st)
        relation_matrix = torch.bmm(relation_matrix, diag)

        H_0 = self.linear_weight0(F_0)
        H_l = self.gcn_layer0(relation_matrix, H_0)
        H_l = self.drop0(H_l)
        H_l = self.gcn_layer1(relation_matrix, H_l)
        H_l = self.drop1(H_l)
        H_l = self.gcn_layer2(relation_matrix, H_l)

        ret = torch.transpose(H_l, 1, 2)
        ret = self.avg_pool(ret)
        ret = torch.transpose(ret, 1, 2)
        ret = ret.contiguous()
        return ret

class SceneEncoder(nn.Module):
    """
    Encoder of Scene Branch.
    """
    def __init__(self, T, d_2D, d_3D, d_model, drop_probability=0.5):
        super(SceneEncoder, self).__init__()
        self.T = T
        self.d_2d = d_2D
        self.d_3d = d_3D
        self.d_model = d_model
        self.drop_probability = drop_probability

        self.w_2d = nn.Sequential(
            nn.Linear(in_features=d_2D, out_features=d_model, bias=False),
            nn.BatchNorm1d(num_features=T),
            nn.ReLU()
        )
        self.w_3d = nn.Sequential(
            nn.Linear(in_features=d_3D, out_features=d_model, bias=False),
            nn.BatchNorm1d(num_features=T),
            nn.ReLU()
        )
        self.w_fuse = nn.Sequential(
            nn.Linear(in_features=d_model * 2, out_features=d_model, bias=False),
            nn.BatchNorm1d(num_features=T),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=drop_probability)

    def forward(self, F_2D, F_3D):
        """

        Args:
            F_2D: (batch_size, T, d_2d)
            F_3D: (batch_size, T, d_3d)

        Returns:
            F_s: (batch_size, T, d_model)
        """
        f_2d = self.w_2d(F_2D)
        f_2d = self.dropout(f_2d)

        f_3d = self.w_3d(F_3D)
        f_3d = self.dropout(f_3d)

        f_s = self.w_fuse(torch.cat([f_2d, f_3d], dim=-1))
        return f_s


if __name__ == '__main__':
    batch_size = 64
    N = 50
    d_2d = 1024
    d_model = 512
    d2d = 2048
    d3d = 1024

    # s_encoder = SceneEncoder(T=10, d_2D=d2d, d_3D=d3d, d_model=512)
    # F_2d = torch.ones((batch_size, 10, d2d))
    # F_3d = torch.ones((batch_size, 10, d3d))
    # out = s_encoder(F_2d, F_3d)
    # print(out.shape)

    G_st = torch.ones((batch_size, N, N))
    F_0 = torch.ones((batch_size, N, d_2d))
    temp_gcn = GCN(in_feature_size=d_2d, out_feature_size=d_model, N=N)
    
    out = temp_gcn(G_st, F_0)
    print(out.shape)
