import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import GCNConv, GATConv, NNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(2023)

class GCN_loop(torch.nn.Module):

    def __init__(self, num_features, embedding_size, gnn_layers, improved, task = 'regression'):
        super(GCN_loop, self).__init__()
        torch.manual_seed(2023)
        self.gnn_layers = gnn_layers
        self.embedding_size = embedding_size

        # GCN layers
        self.initial_conv = GCNConv(num_features, 
                                    embedding_size, 
                                    improved=improved)
        
        self.conv_layers = ModuleList([])
        for _ in range(self.gnn_layers - 1):
            self.conv_layers.append(GCNConv(embedding_size,
                                            embedding_size,
                                            improved=improved))
            
        # Output layer
        self.readout1 = Linear(2*embedding_size, embedding_size)
        self.readout2 = Linear(embedding_size, 1)
        
    def forward(self, x, edge_index, batch_index, edge_weight = None):

        hidden = self.initial_conv(x, edge_index, edge_weight)
        hidden = F.leaky_relu(hidden)

        for i in range(self.gnn_layers-1):
            hidden = self.conv_layers[i](hidden, edge_index, edge_weight)
            hidden = F.leaky_relu(hidden)

        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        
        hidden = self.readout1(hidden)
        hidden = F.leaky_relu(hidden)

        out = self.readout2(hidden)

        return out
        
class NN_model(torch.nn.Module):
    
    def __init__(self, num_node_features, num_edge_features, emb_size):
        super(NN_model, self).__init__()
        torch.manual_seed(2023)

        nn1 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, num_node_features*emb_size), torch.nn.ReLU())
        self.conv1 = NNConv(in_channels=num_node_features, out_channels=emb_size, nn=nn1, aggr='mean')
        
        #nn2 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, emb_size**2), torch.nn.ReLU())
        #self.conv1 = NNConv(in_channels=emb_size, out_channels=emb_size, nn=nn2, aggr='mean')

        self.lin1 = torch.nn.Linear(emb_size*2, emb_size)

        self.lin2 = torch.nn.Linear(emb_size, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        
        hidden = F.relu(self.conv1(x, edge_index, edge_attr))
        
        #hidden = F.relu(self.conv2(hidden, edge_index, edge_attr))

        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch), 
                            gap(hidden, batch)], dim=1)
        

        hidden = F.relu(self.lin1(hidden))
        out = self.lin2(hidden)
        return out


class GAT_loop(torch.nn.Module):
    def __init__(self, num_features, embedding_size, nheads, concat, gnn_layers):
        # Init parent
        super(GAT_loop, self).__init__()
        torch.manual_seed(2023)

        self.gnn_layers = gnn_layers

        # GCN layers
        self.initial_conv = GATConv(num_features, 
                                    embedding_size,
                                    heads=nheads,
                                    concat=concat)
        

        self.conv_layers = ModuleList([])
        for _ in range(self.gnn_layers - 1):
            self.conv_layers.append(GATConv(embedding_size, 
                                            embedding_size,
                                            heads=nheads,
                                            concat=concat))

        # Output layer
        self.readout1 = Linear(2*embedding_size, embedding_size)
        self.readout2 = Linear(embedding_size, 1)
        
    def forward(self, x, edge_index, edge_attr, batch_index):

        # First Conv layer
        hidden = self.initial_conv(x, edge_index, edge_attr)
        hidden = F.leaky_relu(hidden)

        # Other Conv layers
        for i in range(self.gnn_layers-1):
            hidden = self.conv_layers[i](hidden, edge_index, edge_attr)
            hidden = F.leaky_relu(hidden)
          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        hidden = self.readout1(hidden)
        hidden = F.leaky_relu(hidden)

        out = self.readout2(hidden)

        return out
