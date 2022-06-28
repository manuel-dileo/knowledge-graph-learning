import src.data.make_dataset as make_dataset
import src.models.train_model as train_model 
import src.models.predict_model as predict_model

def get_data():
    entities_and_type, triples = make_dataset.set_entities_and_type()
    entity_types_count, entities = make_dataset.get_count(entities_and_type)
    subject_dict, object_dict = make_dataset.get_subject_object(entities, triples, entities_and_type, entity_types_count)

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
    hetero_data = get_data()
    #da scommentare solo per rieseguire il training
    #model_training(hetero_data)
    relation_weights = predict_model.get_relations_weights(hetero_data)
    print(relation_weights)
main()
