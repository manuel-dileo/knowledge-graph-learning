from src.data.graph_modelling.functions import draw_result
import src.data.make_dataset as make_dataset
import src.models.train_model as train_model 
import src.models.predict_model as predict_model
import src.data.graph_modelling.semantic_model_class as semantic_model_class
import src.data.graph_modelling.approximation as approximation
import os
import configparser
import networkx as nx
import pickle

def data_preprocessing(properties = False):
    dataset = make_dataset.MakeDataset()

    triples = dataset.set_entities_and_type(properties)
    new_triples = dataset.clean_triples(triples)

    entity_types_count, property_types_count = dataset.get_count()

    subject_dict, object_dict, properties_and_types = dataset.get_subject_object(new_triples, entity_types_count, property_types_count)

    hetero_data = train_model.create_data(entity_types_count, subject_dict, object_dict, properties_and_types, property_types_count)
    
    return hetero_data

def model_training(hetero_data):
    model, out, optimizer, criterion = train_model.get_model(hetero_data)
    train_model.train_and_save(model, out, optimizer, criterion, hetero_data)

    train_link, val_link, test_link, edge_types =  train_model.split_dataset(hetero_data)
    roc_train = train_model.test_hetlinkpre(model, edge_types, test_link)
    roc_test = train_model.test_hetlinkpre(model, edge_types, test_link)
    print(f'Train AUROC: {roc_train:.4f}\nTest AUROC: {roc_test:.4f}')

def test_data(test_set):
    print("prova")

def main():
    n_experiment = "01"
    path_image = "/home/sara/Desktop/fase2/git_repo/"+\
    "knowledge-graph-learning/data/interim/semantic_models/"+\
    n_experiment+"/confronti/"

    config = configparser.ConfigParser()
    config_path = str(os.path.dirname(os.path.abspath(__file__)))
    config.read(os.path.join(config_path, 'config.ini'))

    path_h = "/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/interim/hetero_data/data.pickle"

    #da scommentare solo per rieseguire il training
    hetero_data = data_preprocessing(False)  
    model_training(hetero_data)

    semantic_model = semantic_model_class.SemanticModelClass()
    sm = semantic_model.parse()
    Uc, Er = semantic_model.algorithm(sm)

    #get closure 
    sd = nx.MultiGraph()
    sdbis = nx.MultiDiGraph()
    for e in Er:
        ebis0 = semantic_model.get_relation_type(e[0])
        ebis2 = semantic_model.get_relation_type(e[2])
        sd.add_node(e[0])
        sd.add_node(e[2])
        sdbis.add_node(ebis0)
        sdbis.add_node(ebis2)

        rel_type = semantic_model.get_relation_type(e[1])
        lw = rel_type + " - " + str(e[3])
        sd.add_edge(e[0],e[2], label = e[1], weight = e[3], lw = lw)
        sdbis.add_edge(ebis0,ebis2, label = rel_type)

    #test_data = #funzione che crea data['tipo'].x e .edge_index per ogni elemento
    #di closure graph + per ogni proprietà
    #relation_weights = predict_model.get_relations_weights(test_data)
    #print(relation_weights)
    semantic_model.draw_result(sdbis, path_image + n_experiment+"_basesdbis")

    semantic_model.draw_result(sd, path_image + n_experiment+"_base")

    relation_weights = predict_model.get_relations_weights(hetero_data)
    print(relation_weights)

    list_closure =[]
    for edge in sd.edges:
        u = edge[0]
        v = edge[1]
        relations = sd.get_edge_data(u,v)
        for i in range(0, len(relations)):
            if (u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]) not in list_closure:
                list_closure.append((u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]))
    
    #for el in list_closure:
    #    print(el)
    
    rgcn_weights_only = semantic_model.update_graph_weights(sd, relation_weights, True)
    both_weights = semantic_model.update_graph_weights(sd, relation_weights, False)
    
    #print("-----------RGCN ONLY-----------------")
    
    rgcn_weights_list = []
    for edge in rgcn_weights_only.edges:
        u = edge[0]
        v = edge[1]
        relations = rgcn_weights_only.get_edge_data(u,v)
        for i in range(0, len(relations)):
            if (u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]) not in rgcn_weights_list:
                rgcn_weights_list.append((u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]))

    #for el in rgcn_weights_list:
        #print(el)


    #print("-----------BOTH-----------------")
    both_list = []
    for edge in both_weights.edges:
        u = edge[0]
        v = edge[1]
        relations = both_weights.get_edge_data(u,v)
        
        for i in range(0, len(relations)):
            if (u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]) not in both_list:
                both_list.append((u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]))
    
    #for el in both_list:
    #   print(el)
    for edge in sd.edges:
        u = edge[0]
        v = edge[1]
        relations = sd.get_edge_data(u,v)
        #print("B ", u, v, relations)

    undirected = sd.to_undirected()

    for edge in undirected.edges:
        u = edge[0]
        v = edge[1]
        relations = undirected.get_edge_data(u,v)
        #print("UNDIR ", u, v, relations)

        
    #STEINER TREE
    ontology_weights_only_tree = approximation.steiner_tree(sd, semantic_model.get_leafs(), weight='weight')
    only_rgcn_tree = approximation.steiner_tree(rgcn_weights_only, semantic_model.get_leafs(), weight='weight')
    both_weights_tree = approximation.steiner_tree(both_weights, semantic_model.get_leafs(), weight='weight')

    #DRAWING CLOSURES
    for edge in sd.edges(data=True): edge[2]['label'] = edge[2]['lw']
    for edge in rgcn_weights_only.edges(data=True): edge[2]['label'] = edge[2]['lw']
    for edge in both_weights.edges(data=True): edge[2]['label'] = edge[2]['lw']

    semantic_model.draw_result(sd, path_image + n_experiment+"_structural")
    semantic_model.draw_result(rgcn_weights_only, path_image + n_experiment+"_rgcn")
    semantic_model.draw_result(both_weights, path_image + n_experiment+ "_both")

    #DRAW RESULT
    for edge in ontology_weights_only_tree.edges(data=True): edge[2]['label'] = edge[2]['lw']
    for edge in only_rgcn_tree.edges(data=True): edge[2]['label'] = edge[2]['lw']
    for edge in both_weights_tree.edges(data=True): edge[2]['label'] = edge[2]['lw']

    semantic_model.draw_result(ontology_weights_only_tree, path_image + n_experiment+ "_structural_tree")
    semantic_model.draw_result(only_rgcn_tree, path_image + n_experiment + "_rgcn_tree")
    semantic_model.draw_result(both_weights_tree, path_image + n_experiment+ "_both_tree")
    
main()