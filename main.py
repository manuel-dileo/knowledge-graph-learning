from src.data.graph_modelling.functions import draw_result
import src.data.make_dataset as make_dataset
import src.models.train_model as train_model 
import src.models.predict_model as predict_model
import src.data.graph_modelling.semantic_model_class as semantic_model_class
import src.data.graph_modelling.approximation as approximation
import os
import configparser
import networkx as nx

def get_data(properties = False):
    dataset = make_dataset.MakeDataset()
    entities_and_type, triples, properties_and_types = dataset.set_entities_and_type(True)
    properties_and_types = dataset.get_classes_types(properties_and_types)
    entity_types_count, entities, property_types_count = dataset.get_count(entities_and_type, properties_and_types)
    subject_dict, object_dict = dataset.get_subject_object(
                                                entities, 
                                                triples, 
                                                entities_and_type, 
                                                entity_types_count,
                                                properties_and_types,
                                                property_types_count
                                                )

    hetero_data = train_model.create_data(entity_types_count, subject_dict, object_dict, properties_and_types, property_types_count)
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

    #semantic_model.draw_result(closure_graph, path_image + "01")
    #hetero_data = get_data(properties = False)

    #da scommentare solo per rieseguire il training
    #model_training(hetero_data)
    #relation_weights = predict_model.get_relations_weights(hetero_data)


    semantic_model = semantic_model_class.SemanticModelClass()
    sm = semantic_model.parse()
    #dist = semantic_model.get_distance("http://dbpedia.org/ontology/Person", "http://dbpedia.org/ontology/BodyDouble")
    #print("_-----------------", dist)
    
    Uc, Er = semantic_model.algorithm(sm)
    closure_graph = nx.MultiDiGraph()
    dict_print = {}
    for e in Er:
        closure_graph.add_node(e[0])
        closure_graph.add_node(e[2])
        rel_type = semantic_model.get_relation_type(e[1])
        lw = rel_type + " - " + str(e[3])
        closure_graph.add_edge(e[0],e[2], label = e[1], weight = e[3], lw = lw)

    semantic_model.draw_result(closure_graph, path_image + "09_test")

    print("-------------ontology_weights_only-----------")

    list_closure =[]
    for edge in closure_graph.edges:
        u = edge[0]
        v = edge[1]
        relations = closure_graph.get_edge_data(u,v)
        for i in range(0, len(relations)):
            if (u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]) not in list_closure:
                list_closure.append((u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]))
    
    
    ontology_weights_only_tree = approximation.steiner_tree(closure_graph.to_undirected(), semantic_model.get_leafs(), weight='weight')
    for edge in ontology_weights_only_tree.edges(data=True): edge[2]['label'] = edge[2]['lw']
    semantic_model.draw_result(ontology_weights_only_tree, path_image + "08_ontology_weights_only_tree")

    #for el in list_closure:
    #    print(el)
    '''
    both_weights = semantic_model.update_graph_weights(closure_graph, relation_weights, False)
    rgcn_weights_only = semantic_model.update_graph_weights(closure_graph, relation_weights, True)
    
    print("-----------RGCN ONLY-----------------")
    
    rgcn_weights_list = []
    for edge in rgcn_weights_only.edges:
        u = edge[0]
        v = edge[1]
        relations = rgcn_weights_only.get_edge_data(u,v)
        for i in range(0, len(relations)):
            if (u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]) not in rgcn_weights_list:
                rgcn_weights_list.append((u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]))

    for el in rgcn_weights_list:
        print(el)


    print("-----------BOTH-----------------")
    both_list = []
    for edge in both_weights.edges:
        u = edge[0]
        v = edge[1]
        relations = both_weights.get_edge_data(u,v)
        
        for i in range(0, len(relations)):
            if (u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]) not in both_list:
                both_list.append((u, relations[i]["label"], v, relations[i]["weight"], relations[i]["lw"]))
    
    for el in both_list:
       print(el)
    
    #STEINER TREE
    ontology_weights_only_tree = approximation.steiner_tree(closure_graph.to_undirected(), semantic_model.get_leafs(), weight='weight')
    only_rgcn_tree = approximation.steiner_tree(rgcn_weights_only.to_undirected(), semantic_model.get_leafs(), weight='weight')
    both_weights_tree = approximation.steiner_tree(both_weights.to_undirected(), semantic_model.get_leafs(), weight='weight')


    #DRAWING CLOSURES
    for edge in closure_graph.edges(data=True): edge[2]['label'] = edge[2]['lw']
    for edge in rgcn_weights_only.edges(data=True): edge[2]['label'] = edge[2]['lw']
    for edge in both_weights.edges(data=True): edge[2]['label'] = edge[2]['lw']

    semantic_model.draw_result(closure_graph, path_image + "06_ontology_weights_only")
    semantic_model.draw_result(rgcn_weights_only, path_image + "06_rgcn_weights_only")
    semantic_model.draw_result(both_weights, path_image + "06_both_weights")

    #DRAW RESULT
    for edge in ontology_weights_only_tree.edges(data=True): edge[2]['label'] = edge[2]['lw']
    for edge in only_rgcn_tree.edges(data=True): edge[2]['label'] = edge[2]['lw']
    for edge in both_weights_tree.edges(data=True): edge[2]['label'] = edge[2]['lw']

    semantic_model.draw_result(ontology_weights_only_tree, path_image + "06_ontology_weights_only_tree")
    semantic_model.draw_result(only_rgcn_tree, path_image + "06_only_rgcn_tree")
    semantic_model.draw_result(both_weights_tree, path_image + "06_both_weights_tree")
    '''
main()