from src.data.graph_modelling.functions import draw_result
import src.data.make_dataset as make_dataset
import src.models.train_model as train_model 
import src.models.predict_model as predict_model
import src.data.graph_modelling.semantic_model_class as semantic_model_class
import src.data.graph_modelling.approximation as approximation
import os
import configparser
import networkx as nx

def get_data():
    dataset = make_dataset.MakeDataset()
    entities_and_type, triples = dataset.set_entities_and_type()
    entity_types_count, entities = dataset.get_count(entities_and_type)
    subject_dict, object_dict = dataset.get_subject_object(entities, triples, entities_and_type, entity_types_count)

    hetero_data = train_model.create_data(entity_types_count, subject_dict, object_dict)
    return hetero_data

def model_training(hetero_data):
    model, out, optimizer, criterion = train_model.get_model(hetero_data)
    train_model.train_and_save(model, out, optimizer, criterion, hetero_data)

    train_link, val_link, test_link, edge_types =  train_model.split_dataset(hetero_data)
    roc_train = train_model.test_hetlinkpre(model, edge_types, test_link)
    roc_test = train_model.test_hetlinkpre(model, edge_types, test_link)
    print(f'Train AUROC: {roc_train:.4f}\nTest AUROC: {roc_test:.4f}')

def main():
    path_image = "/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/interim/semantic_models/"
    config = configparser.ConfigParser()
    config_path = str(os.path.dirname(os.path.abspath(__file__)))
    config.read(os.path.join(config_path, 'config.ini'))

    semantic_model = semantic_model_class.SemanticModelClass()
    sm = semantic_model.parse()
    #dist = semantic_model.get_distance("http://dbpedia.org/ontology/Person", "http://dbpedia.org/ontology/BodyDouble")
    #print("_-----------------", dist)
    
    Uc, Er = semantic_model.algorithm(sm)
    closure_graph = nx.MultiDiGraph()
    for e in Er:
        closure_graph.add_node(e[0])
        closure_graph.add_node(e[2])
        closure_graph.add_edge(e[0],e[2], label = e[1], weight = e[3])
        #print(e)
   
    semantic_model.draw_result(closure_graph, path_image + "04")
    '''

    hetero_data = get_data()
    #da scommentare solo per rieseguire il training
    #model_training(hetero_data)
    relation_weights = predict_model.get_relations_weights(hetero_data)
    
    print("-------------ontology_weights_only-----------")
    for edge in closure_graph.edges:
        u = edge[0]
        v = edge[1]
        rel = closure_graph.get_edge_data(u,v)[0]
        print(u, rel["label"], v, rel["weight"])

    both_weights = semantic_model.update_graph_weights(closure_graph, relation_weights, False)
    rgcn_weights_only = semantic_model.update_graph_weights(closure_graph, relation_weights, True)
    
    print("-----------RGCN ONLY-----------------")
    for edge in rgcn_weights_only.edges:
        u = edge[0]
        v = edge[1]
        rel = rgcn_weights_only.get_edge_data(u,v)[0]
        print(u, rel["label"], v, rel["weight"])

    print("-----------BOTH-----------------")
    for edge in both_weights.edges:
        u = edge[0]
        v = edge[1]
        rel = both_weights.get_edge_data(u,v)[0]
        print(u, rel["label"], v, rel["weight"])

    #STEINER TREE
    ontology_weights_only = approximation.steiner_tree(closure_graph.to_undirected(), semantic_model.get_leafs(), weight='weight')
    only_rgcn = approximation.steiner_tree(rgcn_weights_only.to_undirected(), semantic_model.get_leafs(), weight='weight')
    both_weights_tree = approximation.steiner_tree(both_weights.to_undirected(), semantic_model.get_leafs(), weight='weight')

    #DRAW RESULT
    semantic_model.draw_result(ontology_weights_only, path_image + "06_ontology_weights_only")
    semantic_model.draw_result(only_rgcn, path_image + "06_only_rgcn")
    semantic_model.draw_result(both_weights_tree, path_image + "06_both_weights_tree")
    '''
main()