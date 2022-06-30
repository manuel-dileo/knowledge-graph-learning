from src.data.graph_modelling.functions import draw_result
import src.data.make_dataset as make_dataset
import src.models.train_model as train_model 
import src.models.predict_model as predict_model
import src.data.graph_modelling.semantic_model_class as semantic_model_class
import src.data.graph_modelling.approximation as approximation
import os
import configparser

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
    path_image = "/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/graph_images/"
    config = configparser.ConfigParser()
    config_path = str(os.path.dirname(os.path.abspath(__file__)))
    config.read(os.path.join(config_path, 'config.ini'))

    #hetero_data = get_data()
    #da scommentare solo per rieseguire il training
    #model_training(hetero_data)
    #relation_weights = predict_model.get_relations_weights(hetero_data)
    
    semantic_model = semantic_model_class.SemanticModelClass()
    sm = semantic_model.parse()
    #closure = semantic_model.compute_closure_graph(sm)
    #semantic_model.draw_result(closure, path_image + "closure_node")
    closure = semantic_model.algorithm(sm)
    semantic_model.draw_result(closure, path_image + "closure_node")

    #closure_graph = semantic_model.compute_closure_graph(sm)
    
    #new_closure = semantic_model.set_graph_weights(closure_graph, relation_weights)

    #new_closure = semantic_model.update_graph_weights(closure_graph, relation_weights)
    #ontology_weights_only = approximation.steiner_tree(closure_graph.to_undirected(), semantic_model.get_leafs(), weight='weight')
    #only_rgcn = approximation.steiner_tree(new_closure.to_undirected(), semantic_model.get_leafs(), weight='weight')

    #semantic_model.draw_result(new_closure, path_image + "new_closure")
    #semantic_model.draw_result(ontology_weights_only, path_image + "ontology_weights_only")
    #print(semantic_model.graph_to_json(new_closure))

main()