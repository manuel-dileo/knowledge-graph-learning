from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import SAGEConv, to_hetero, GATConv
import configparser
import os
config = configparser.ConfigParser()
config_path=str(os.path.dirname(os.path.abspath(__file__))).split(os.sep)
config_path = "/".join(config_path[0:len(config_path)-2])
config.read(os.path.join(config_path, 'config.ini'))

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        #Sto definendo quale tipologia di layer voglio usare.
        self.conv1 = GATConv((-1, -1), hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        #descrive la computazione dall'input all'output.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def set_model(data):
    model = GNN(hidden_channels=4, out_channels=2)
    model = to_hetero(model, data.metadata(), aggr='sum')
    #model.load_state_dict(torch.load(config_path+config['model']['path']))
    path = '/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/models/model_weigths.pth'
    model.load_state_dict(torch.load(path))
    return model

def test_hetscores(model, test_link):
    model.eval()
    out = model(test_link.x_dict, test_link.edge_index_dict)
    
    ### LINK PREDICTION ACTS HERE ###
    
    hs = torch.Tensor()
    ### LINK PREDICTION ACTS HERE ###
    for edge_t in test_link.edge_index_dict.keys():
        #Compute link embedding for each edge type
        #for src in train_link[edge_t].edge_label_index[0]:
        out_src = out[edge_t[0]][test_link[edge_t].edge_index[0]]#embedding src nodes
        out_dst = out[edge_t[2]][test_link[edge_t].edge_index[1]] #embedding dst nodes
        
        # LINK EMBEDDING #
        # 1 - Dot Product
        out_sim = out_src * out_dst #dotproduct
        h = torch.sum(out_sim, dim=-1)
        
        hs = torch.cat((hs,h),-1)
    
    
    pred_cont = torch.sigmoid(hs).cpu().detach().numpy()
    
    return pred_cont

def get_relations_weights(test_data, hetero):
    model = set_model(hetero)
    #nella sd non possono esserci archi mai visti nel training
    #weight = test_hetscores(model, hetero)[0]
    data = HeteroData()
    relations_weights={}
    for triple in test_data.edge_index_dict.keys():
        for triple2 in test_data.edge_index_dict.keys():
            data[triple2].edge_index = torch.Tensor([[],[]]).long()
            data[triple2[0]].x = torch.Tensor([[1]])
            data[triple2[2]].x = torch.Tensor([[1]])
        data[triple[0]].x = torch.Tensor([[1]])
        data[triple[2]].x = torch.Tensor([[1]])
        data[triple].edge_index = torch.Tensor([[0],[0]]).long()
        weight = test_hetscores(model, data)[0]
        relations_weights[triple] = weight
        #print(f'{triple}: {relations_weights}')
    
    return relations_weights
    
