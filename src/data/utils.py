from torch_geometric.data import HeteroData
import torch
import numpy as np


def triples_to_heterodata(Er, Et):
    data = HeteroData()
    entity_count = _entities_count(Er)
    subject_dict, object_dict = get_dicts(Er, entity_count)
    #property_count = _properties_count(Et)

    for type in entity_count.keys():
        data[type] = [[1] for i in range(entity_count[type])]

    for triple in subject_dict.keys(): 
        lol = [subject_dict[triple], object_dict[triple]]
        data[triple[0], triple[1], triple[2]].edge_index = torch.Tensor(lol).long()
    return data

def get_dicts(Er, entity_count, property_count = {}):
    subject_dict = {}
    object_dict = {}
    
    index_dict = {t:{'count': 0} for t in entity_count.keys()}
    '''
    for class_name, subject,rel, p_type in property_count.keys():
        index_dict[p_type] = {'count':0}
        if class_name not in index_dict.keys():
            index_dict[class_name] = {'count':0}  
    '''
    for s,p,o, weight in Er:
        s_type = get_type(s)
        p_type = get_type(p)
        o_type = get_type(o)
        s_type = s_type[0:len(s_type)-1]
        o_type = o_type[0:len(o_type)-1]


        if(s_type != "" and o_type != ""):
            key_t = (s_type, p_type, o_type)

        if key_t not in subject_dict.keys():
            subject_dict[key_t] = []
            object_dict[key_t] = []

        if s[0:len(s)-1] not in index_dict[s_type]:
            index_dict[s_type][s] = index_dict[s_type]['count']
            index_dict[s_type]['count'] = index_dict[s_type]['count']+1
        s_index = index_dict[s_type][s]
        
        if o[0:len(o)-1] not in index_dict[o_type]:
            index_dict[o_type][o] = index_dict[o_type]['count']
            index_dict[o_type]['count'] = index_dict[o_type]['count']+1
        o_index = index_dict[o_type][o]
                
        subject_dict[key_t].append(s_index)
        object_dict[key_t].append(o_index)
    return (subject_dict, object_dict)

def _entities_count(Er):
    entities_count = {}
    for s, p, o, w in Er:
        s_type = get_type(s)
        o_type = get_type(o)
        s_type = s_type[0:len(s_type)-1]
        o_type = o_type[0:len(o_type)-1]
        entities_count[s_type] = entities_count.get(s_type, 0) +1
        entities_count[o_type] = entities_count.get(o_type, 0) +1
    return entities_count


def _properties_count(Er):

    
    properties_count = {}
    return properties_count

def get_type( relation):
    r_split = relation.split("/")
    return r_split[len(r_split)-1]

def get_property_type( property):
    split_p = property.split("^^")
    p_type = str(split_p[1].split("#")[1]).lower()
    
    if p_type.startswith("xsd:integer"):
        return("Integer", split_p[0])
    if p_type.startswith("xsd:string"):
        return("String", split_p[0])
    if p_type.startswith("xsd:double"):
        return("Double", split_p[0])
    if p_type.startswith("xsd:gYear"):
        return("Year",split_p[0])
    if p_type.startswith("xsd:date"):
        return("Date",split_p[0])
    return ("","")

def get_ontology_type( property):
    p_type = str(property.split("#")[1]).lower()
    
    if p_type.startswith("int"):
        return("Integer")
    if p_type.startswith("string"):
        return("String")
    if p_type.startswith("double") or p_type.startswith("decimal"):
        return("Double")
    if p_type.startswith("gYear"):
        return("Year")
    if p_type.startswith("date"):
        return("Date")
    return ("")
