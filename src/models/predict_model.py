from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import SAGEConv, to_hetero, GATConv

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
    model.load_state_dict(torch.load('git_repo/knowledge-graph-learning/models/model_weigths.pth'))
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

def get_relations_weights(data):
    test_data = HeteroData()
    model = set_model(data)

    relations_weights={}
    for triple in data.edge_index_dict.keys():
        for triple2 in data.edge_index_dict.keys():
            test_data[triple2].edge_index = torch.Tensor([[],[]]).long()
            test_data[triple2[0]].x = torch.Tensor([[1]])
            test_data[triple2[2]].x = torch.Tensor([[1]])
        test_data[triple[0]].x = torch.Tensor([[1]])
        test_data[triple[2]].x = torch.Tensor([[1]])
        test_data[triple].edge_index = torch.Tensor([[0],[0]]).long()
        weight = test_hetscores(model, test_data)[0]
        relations_weights[triple] = weight
        #print(f'{triple}: {relations_weights}')
    return relations_weights