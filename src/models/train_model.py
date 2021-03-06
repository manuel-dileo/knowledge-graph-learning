import torch
from re import I
from sklearn.linear_model import OrthogonalMatchingPursuit
from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import SAGEConv, to_hetero, GATConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
import numpy as np 
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, GATConv
import configparser
import os
from langdetect import detect
from dateutil.parser import parse
import datetime, string
from torch_geometric.utils import remove_self_loops, remove_isolated_nodes

config = configparser.ConfigParser()
config_path=str(os.path.dirname(os.path.abspath(__file__))).split(os.sep)
config_path = "/".join(config_path[0:len(config_path)-2])
config.read(os.path.join(config_path, 'config.ini'))

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def get_model(data):
    model = GNN(hidden_channels=4, out_channels=2)
    model = to_hetero(model, data.metadata(), aggr='sum')

    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict,data.edge_index_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion =  torch.nn.BCEWithLogitsLoss() #change loss function

    return model, out, optimizer, criterion
    
def create_data(entity_types_count, subject_dict, object_dict, properties_and_types = {}, property_types_count = {}):
    data = HeteroData()

    data_to_insert = {}
    for subj in list(properties_and_types.keys()):
        for class_type, prop_name, prop_type, prop_value in properties_and_types[subj]:
            if prop_type not in data_to_insert:
                data_to_insert[prop_type] = []
            else:
                for i in range(property_types_count[(class_type, subj, prop_name, prop_type)]):
                    data_to_insert[prop_type].append(function_build_feature(prop_type, prop_value))


    types = list(entity_types_count.keys())
    for t in types:
        data_to_insert[t] = [[1] for i in range(entity_types_count[t])]

    for key in data_to_insert.keys():
        lists = data_to_insert[key]
        if lists != '':
            data[key].x = torch.Tensor(lists)
            
    for triple in subject_dict.keys():  #le keys di subject_dict ed object_dict sono ==
        #ci sono 2 dizionari diversi per evitare il doppio for (complessit?? n^2)
        lol = [subject_dict[triple], object_dict[triple]]
        data[triple[0], triple[1], triple[2]].edge_index = torch.Tensor(lol).long()

    
    for k,v in data.edge_index_dict.items():
        max_ind_sx = int(max(v[1]))
        try:
            n1 = data[k[2]].x[max_ind_sx]
        except IndexError:
            print("Relation:", k, " node type:", k[2], " index:", max_ind_sx, f"{k[2]} matrix dimension:", len(complete_data[k[2]].x))
        
    
    return data

def remove_isolated_nodes(data):

    for edge_type in data.edge_index_dict.keys():
        if edge_type[0] == edge_type[2]:
            new_edge_index = remove_self_loops(data[edge_type].edge_index)[0]
            data[edge_type].edge_index = new_edge_index
        new_edge_index = remove_isolated_nodes(data[edge_type].edge_index)[0]
        data[edge_type].edge_index = new_edge_index
    return data

def split_dataset(data):

    edge_types = list(data.edge_index_dict.keys())

    link_split = RandomLinkSplit(num_val=0.0,
                                num_test=0.25,
                                edge_types=edge_types,
                                rev_edge_types=[None]*len(edge_types))
    train_link, val_link, test_link = link_split(data)
    return train_link, val_link, test_link, edge_types

def train_hetlinkpre(model, out, optimizer, criterion, data):
    train_link, val_link, test_link, edge_types = split_dataset(data)

    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(train_link.x_dict, train_link.edge_index_dict)  # Perform a single forward pass.
    preds = torch.Tensor()
    edge_labels = torch.Tensor()
    ### LINK PREDICTION ACTS HERE ###
    for edge_t in edge_types:
        #Compute link embedding for each edge type
        #for src in train_link[edge_t].edge_label_index[0]:
        out_src = out[edge_t[0]][train_link[edge_t].edge_label_index[0]]#embedding src nodes
        out_dst = out[edge_t[2]][train_link[edge_t].edge_label_index[1]] #embedding dst nodes
        
        # LINK EMBEDDING #
        # 1 - Dot Product
        out_sim = out_src * out_dst #dotproduct
        pred = torch.sum(out_sim, dim=-1)
        
        preds = torch.cat((preds,pred),-1)
        edge_labels = torch.cat((edge_labels,train_link[edge_t].edge_label.type_as(pred)),-1)
    
        
    #compute loss function based on all edge types
    loss = criterion(preds, edge_labels)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss



def test_hetlinkpre(model, edge_types, test_link):
    model.eval()
    out = model(test_link.x_dict, test_link.edge_index_dict)
    
    ### LINK PREDICTION ACTS HERE ###
    
    hs = torch.Tensor()
    edge_labels = np.array([])
    ### LINK PREDICTION ACTS HERE ###
    for edge_t in edge_types:
        #Compute link embedding for each edge type
        #for src in train_link[edge_t].edge_label_index[0]:
        out_src = out[edge_t[0]][test_link[edge_t].edge_label_index[0]]#embedding src nodes
        out_dst = out[edge_t[2]][test_link[edge_t].edge_label_index[1]] #embedding dst nodes
        
        # LINK EMBEDDING #
        # 1 - Dot Product
        out_sim = out_src * out_dst #dotproduct
        h = torch.sum(out_sim, dim=-1)
        
        hs = torch.cat((hs,h),-1)
        edge_labels = np.concatenate((edge_labels,test_link[edge_t].edge_label.cpu().detach().numpy()))
    
    
    pred_cont = torch.sigmoid(hs).cpu().detach().numpy()
    
    # EVALUATION
    test_roc_score = roc_auc_score(edge_labels, pred_cont)#, pred_function_build_feature
    return test_roc_score

def train_and_save(model, out, optimizer, criterion, data):
    for epoch in range(1,1001):
        loss = train_hetlinkpre(model, out, optimizer, criterion, data)
    path="/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/models/"
    #torch.save(model.state_dict(), config['model']['path'])
    torch.save(model.state_dict(), config_path + config['model']['path'])

def function_build_feature(p_type, value):
    #return [5] cos?? funziona perch?? ?? numerico
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))

    #aggiungere funzione x riconscere le date
    if p_type == 'Integer':
        try: i = int(value) 
        except: i = 0
        return [i]
    if p_type == 'Double':
        try: d = float(value)
        except: d = float(0.0)
        return [d]
    if p_type == 'gYear':
        return [int(1970-value)]
    if p_type == 'String':
        a_punct = count(value, string.punctuation)
        lang = 0
        try:
            if detect(value) == 'en': lang = 1
        except:
            lang = 0
        return [len(value), value.count(" ") , value.count("(") + value.count(")"), lang, a_punct]
    if p_type == 'Date':
        return [(parse(value) - datetime.datetime(1970,1,1)).days]
    return ""

def get_type(self, relation):
    r_split = relation.split("/")
    return r_split[len(r_split)-1]

