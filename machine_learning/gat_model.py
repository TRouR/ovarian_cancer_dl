import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils import activation_fun

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) model with configurable attention and fully connected layers.

    Args:
        input_dim (int): Dimensionality of input features.
        label_dim (int): Number of output classes.
        num_features (int): Number of graph nodes (used for pooling).
        params (dict): Dictionary of hyperparameters for model configuration.
    """
    def __init__(self, input_dim, label_dim, num_features, params):
        super(GAT, self).__init__()

        self.dropout_gat = params['dropout_gat']
        self.dropout_fc = params['dropout_fc']
        self.dropout_att = params['dropout_att']

        self.num_gat_layers = params['num_gat_layers']
        self.num_fc_layers = params['num_fc_layers']
        self.normalization = params['normalization']
        self.which_layer = params['which_layer']

        self.act_fc = activation_fun(params['act_fc'])
        self.act_gat = activation_fun(params['act_gat_out'])
        self.act_att = activation_fun(params['act_attention'], negative_slope=params['negative_slope'])

        self.nhids = [params[f'nhids_{i}'] for i in range(1, self.num_gat_layers + 1)]
        self.nheads = [params[f'nheads_{i}'] for i in range(1, self.num_gat_layers + 1)]
        self.fc_dims = [params[f'fc_dim_{i}'] for i in range(1, self.num_fc_layers + 1)]

        self.attentions = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_dim = input_dim
        for i in range(self.num_gat_layers):
            heads = nn.ModuleList([
                GraphAttentionLayer(prev_dim, self.nhids[i], self.dropout_att,
                                     self.act_gat, self.act_att) for _ in range(self.nheads[i])
            ])
            self.attentions.append(heads)
            self.pools.append(nn.Linear(self.nhids[i] * self.nheads[i], 1))
            prev_dim = self.nhids[i] * self.nheads[i]

        self.dropout_layer = nn.Dropout(p=self.dropout_gat)

        self.lin_input_dim = (self.num_gat_layers + 1) * num_features if self.which_layer == 'all' else num_features
        self._build_fc_layers()
        self.classifier = nn.Linear(self.fc_dims[-1], label_dim)

    def _build_fc_layers(self):
        layers = []
        prev_dim = self.lin_input_dim

        for fc_dim in self.fc_dims:
            fc_layer = nn.Linear(prev_dim, fc_dim)

            # Custom weight initialization
            if isinstance(self.act_fc, nn.ReLU):
                init.kaiming_uniform_(fc_layer.weight, mode='fan_in', nonlinearity='relu')
            else:
                init.kaiming_uniform_(fc_layer.weight, a=1.0, mode='fan_in', nonlinearity='leaky_relu')
            init.zeros_(fc_layer.bias)

            # Construct layer
            layer = [fc_layer]
            if self.normalization == 'batch':
                layer.append(nn.BatchNorm1d(fc_dim))
            elif self.normalization == 'layer':
                layer.append(nn.LayerNorm(fc_dim))

            layer += [self.act_fc, nn.Dropout(self.dropout_fc)]
            layers.append(nn.Sequential(*layer))

            prev_dim = fc_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, x, adj, labels, which_layer):
        # Aggregate node-level features using GAT layers
        x0 = torch.mean(x, dim=-1)
        x_list, attn_list = [x0], []

        for i, heads in enumerate(self.attentions):
            x = self.dropout_layer(x)
            outputs, attns = zip(*[head(x, adj) for head in heads])
            x = torch.cat(outputs, dim=-1)
            attention = torch.stack(attns, dim=-1).mean(dim=-1).unsqueeze(-1)
            pooled = self.pools[i](x).squeeze(-1)
            x_list.append(pooled)
            attn_list.append(attention)

        x = torch.cat(x_list, dim=-1) if which_layer == 'all' else x_list[-1]
        attention_map = torch.cat(attn_list, dim=-1).mean(dim=-1)

        gat_features = x
        #print("EDED",x.size())
        fc_features = self.encoder(x)
        logits = self.classifier(fc_features)

        return gat_features, fc_features, logits, attention_map
    

class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention Layer as used in the Graph Attention Network (GAT).

    Args:
        in_features (int): Input feature dimension per node.
        out_features (int): Output feature dimension per node.
        dropout (float): Dropout rate applied to attention coefficients.
        act_gat (callable): Activation applied after aggregation (e.g., ReLU).
        act_att (callable): Activation applied to attention scores (e.g., LeakyReLU).
        act_type_gat (str): Type of activation for weight init selection.
        concat (bool): If True, applies non-linearity to the output; else linear.
    """
    def __init__(self, in_features, out_features, dropout, act_gat, act_att, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act_gat = act_gat
        self.act_att = act_att
        self.concat = concat

        # Learnable parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        # Weight initialization
        if isinstance(self.act_gat, nn.ReLU):
            init.kaiming_uniform_(self.W.data, mode='fan_in', nonlinearity='relu')
        else:  # 'Elu' assumed
            init.kaiming_uniform_(self.W.data, a=1.0, mode='fan_in', nonlinearity='leaky_relu')

        init.kaiming_uniform_(self.a.data, a=self.act_att.negative_slope, mode='fan_in', nonlinearity='leaky_relu')

        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, input, adj):
        """
        Forward pass for GAT layer.

        Args:
            input (Tensor): Node features [batch_size, N, in_features].
            adj (Tensor): Binary adjacency matrix [N, N].

        Returns:
            output (Tensor): Transformed node features [batch_size, N, out_features].
            attention (Tensor): Attention weights [batch_size, N, N, 1].
        """
        h = torch.matmul(input, self.W)  # [B, N, F]
        B, N, _ = h.shape

        a_input = torch.cat([
            h.repeat(1, 1, N).view(B, N * N, -1),
            h.repeat(1, N, 1)
        ], dim=-1).view(B, N, N, 2 * self.out_features)

        e = self.act_att(torch.matmul(a_input, self.a).squeeze(-1))  # [B, N, N]
        batch_adj = adj.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(batch_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)

        h_prime = torch.bmm(attention, h)  # [B, N, F]
        attention = attention.unsqueeze(-1)  # [B, N, N, 1]
        
        return self.act_gat(h_prime) if self.concat else h_prime, attention

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_features} → {self.out_features})'
