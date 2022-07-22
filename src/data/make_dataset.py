from asyncio import proactor_events
from tkinter import S
import rdflib
from rdflib import URIRef, RDF
from rdflib.namespace import Namespace
import configparser
import os


class MakeDataset():
    def __init__(self):

        config_path=str(os.path.dirname(os.path.abspath(__file__))).split(os.sep)
        config_path = "/".join(config_path[0:len(config_path)-2])
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(config_path, 'config.ini'))
        self.possible_types = {}
        self.ontology = rdflib.Graph()
        self.ontology.parse(self.config['ontology']['path'], format=self.config['ontology']['format'])
        self.ontology.bind("dbo", Namespace("http://dbpedia.org/ontology/"))
        self.ontology.bind("dbr", Namespace("http://dbpedia.org/resource/"))
        self.ontology.bind("rdfs", Namespace("http://www.w3.org/2000/01/rdf-schema#"))
        self.ontology.bind("owl", Namespace("http://www.w3.org/2002/07/owl#"))
        self.ontology.bind("rdf", Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"))


    def get_property_type(self, property):
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

    def get_type(self, relation):
        r_split = relation.split("/")
        return r_split[len(r_split)-1]


    def get_possible_types(self,subj_type, obj_type):
        if (subj_type,obj_type) not in self.possible_types:
            q = "SELECT DISTINCT ?property WHERE {"+\
            "{ ?property rdfs:domain dbo:"+subj_type+". ?property rdfs:range dbo:"+obj_type+\
            " .} UNION {dbo:"+subj_type +" rdfs:subClassOf ?superclass. dbo:"+obj_type +" rdfs:subClassOf  ?superclass2 ."+\
            "  ?property rdfs:domain ?superclass . ?property rdfs:range ?superclass2 "+\
            "} }"
            result = self.ontology.query(q)
            results = []
            for res in result:
                results.append(str(res[0]))
            
            q2 = "SELECT DISTINCT ?property WHERE {"+\
            "{dbo:"+subj_type +" rdfs:subClassOf ?superclass. "+\
            " ?property rdfs:domain ?superclass . ?property rdfs:range dbo:"+obj_type+\
            " .} UNION {dbo:"+obj_type +" rdfs:subClassOf  ?superclass2 . ?property rdfs:domain dbo:"+\
            subj_type+" . ?property rdfs:range ?superclass2}}"
            
            result = self.ontology.query(q2)
            for res in result:
                results.append(str(res[0]))
            
            self.possible_types[(subj_type,obj_type)] = results
            return results
        return self.possible_types[(subj_type,obj_type)]

    def get_class_from_property(self):
        Q = " SELECT ?property ?class WHERE {?property rdfs:domain ?class; rdf:type owl:DatatypeProperty. }"
        results = {}
        result = self.ontology.query(Q)
        for res in result:
            results[self.get_type(str(res[0]))]= self.get_type(str(res[1]))
        return results
    
    def get_classes_types(self, properties_and_types):
        class_from_property = self.get_class_from_property()
        new_properties_and_types = {}
        for s in list(properties_and_types.keys()):
            for element in properties_and_types[s]:
                prop = element[0] 
                p_type = element[1]
                p_value = element[2] 
                
                if prop in class_from_property:
                    if s not in new_properties_and_types:
                        new_properties_and_types[s] = []
                    new_properties_and_types[s].append((class_from_property[prop], prop, p_type, p_value))
        
        return new_properties_and_types

    def disambiguate_multiple_types(self, entities_and_type, s,p,o, properties_and_type = {}): 
        
        for subtype_subj in entities_and_type[str(s)]:
            if subtype_subj in list(properties_and_type.keys()):
                continue
            if len(entities_and_type[str(o)]) > 1:
                for subtype_obj in entities_and_type[str(o)]:
                    possible_rels = self.get_possible_types( subtype_subj, subtype_obj)
                    if len(possible_rels) == 0:
                        return ("","")    
                    for rel in possible_rels:
                        if rel == p:
                            return (subtype_subj, subtype_obj)
            else:
                possible_rels = self.get_possible_types( subtype_subj, entities_and_type[str(o)][0])
                if len(possible_rels) == 0:
                        return ("","")    
                for rel in possible_rels:
                    if rel == p:
                        return (subtype_subj, subtype_obj)
                
            return ("","")    

    def get_count(self, entities_and_type, properties_and_types = {}):
        entity_types_count = {}
        property_types_count = {}
        entities = []
        for entity in entities_and_type.keys():
            tipo = entities_and_type[entity][0]
            if tipo != "":
                entity_types_count[tipo] = entity_types_count.get(tipo, 0)+1
                entities.append(entity)

        for subj in properties_and_types.keys():
            for class_name, prop_name, prop_type, prop_value in properties_and_types[subj]:
                property_types_count[(class_name, subj, prop_name, prop_type)] = property_types_count.get((class_name, subj, prop_name,prop_type), 0)+1
        return entity_types_count, entities, property_types_count

    def clean_triples(self, triples, entities_and_type, properties_and_type = {}):
        new_triples = []
        added_types = []

        for s,p,o in triples:
            s1 = str(s)
            p1 = str(p)
            o1 = str(o)
            insert_type = True

            if p != str(RDF.type):
                if str(s) in list(properties_and_type.keys()):
                    #x = properties_and_type[str(s)]
                    new_triples.append((s, p, o))
                if str(s) in list(entities_and_type.keys()) and str(o) in list(entities_and_type.keys()):
                    if len(entities_and_type[str(s)]) > 1:
                        new_subj_type, new_obj_type = self.disambiguate_multiple_types(entities_and_type,s,p,o, properties_and_type)
                        if(new_subj_type, new_obj_type) == ("",""):
                            continue
                        #print("news", new_subj_type, "newo", new_obj_type, "sub", s, "obj", o)
                        if new_subj_type != "" and new_obj_type != "":
                            if s not in added_types:
                                new_triples.append((s, str(RDF.type),self.config["prefixes"]["ontology"]+ new_subj_type[0] ))
                                added_types.append(s)
                            if o not in added_types:
                                new_triples.append((o,str(RDF.type),self.config["prefixes"]["ontology"]+ new_obj_type[0] ))
                                added_types.append(o)
                            new_triples.append((new_subj_type, p, new_obj_type))
                        insert_type = False
                    else: 
                        if s not in added_types:
                            new_triples.append((s, str(RDF.type),self.config["prefixes"]["ontology"]+entities_and_type[str(s)][0] ))
                            added_types.append(s)
                        if o not in added_types and str(o).find("^^") == -1:
                            new_triples.append((o, str(RDF.type),self.config["prefixes"]["ontology"]+entities_and_type[str(o)][0] ))
                            added_types.append(o)
                        new_triples.append((s, p, o))
                    insert_type = False
            else:
                if insert_type:
                    new_triples.append((s, p, o))
                    insert_type = True
        return new_triples

   
    def get_subject_object(self, entities, 
                                triples, 
                                entities_and_type, 
                                entity_types_count, 
                                properties_and_type = {},
                                property_types_count = {}):
        subject_dict = {}
        object_dict = {}

        new_triples= self.clean_triples(triples, entities_and_type, properties_and_type)

        index_dict = {t:{'count': 0} for t in entity_types_count.keys()}

        for class_name, subject,rel, p_type in property_types_count.keys():
            index_dict[p_type] = {'count':0}
            if class_name not in index_dict.keys():
                index_dict[class_name] = {'count':0}  
        new_triples.sort()
        for triple in new_triples:
            s = str(triple[0])
            p = str(triple[1])
            o = str(triple[2])
            type_triples = []
            s_type = entities_and_type[s][0] 
            
            if p != str(RDF.type):
                if o.find("^^") == -1:
                    p_type = self.get_type(p)
                    o_type = entities_and_type[o][0]
                    type_triples.append((s_type,p_type, o_type))

                else: 
                    for properties in properties_and_type[s]:
                        s_type = properties[0]
                        p_type = self.get_type(properties[1])
                        o_type = properties[2]
                        type_triples.append((s_type,p_type, o_type))

                for s_type,p_type,o_type in type_triples:
                    if(s_type != "" and o_type != ""):
                        key_t = (s_type, p_type, o_type)
                        
                        if key_t not in subject_dict.keys():
                            subject_dict[key_t] = []
                            object_dict[key_t] = []
                            
                        if str(s) not in index_dict[s_type]:
                            index_dict[s_type][str(s)] = index_dict[s_type]['count']
                            index_dict[s_type]['count'] = index_dict[s_type]['count']+1
                        s_index = index_dict[s_type][str(s)]
                            
                        if str(o) not in index_dict[o_type]:
                            index_dict[o_type][str(o)] = index_dict[o_type]['count']
                            index_dict[o_type]['count'] = index_dict[o_type]['count']+1
                        o_index = index_dict[o_type][str(o)]
                            
                        subject_dict[key_t].append(s_index)
                        object_dict[key_t].append(o_index)
                
                #data[s_type, p_type, o_type].edge_index[0].append(entities.index(str(s)))
                #data[s_type, p_type, o_type].edge_index[1].append(entities.index(str(o)))
        return subject_dict, object_dict

    def set_entities_and_type(self, use_properties = False):
        entities_and_type = {}
        properties_and_types ={}
        relations = []
        triples = []
        # Process the Knowledge Graph
        g = rdflib.Graph()
        g.parse(self.config["kg"]["path"], format=self.config["kg"]["format"])
        for s, p, o in g:
            str_s = str(s)
            str_p = str(p)
            str_o = str(o)
            if str_p != str(RDF.type):
                if not str_s in entities_and_type.keys():
                    entities_and_type[(str_s)] =[]
                if not str_p in relations:
                    relations.append(str_p)

                if str_o.find('^^') == -1:
                    if not str_o in entities_and_type.keys():
                        entities_and_type[str_o]=[]
                    triples.append((str_s,str_p,str_o))
                else:
                    if use_properties:
                        if str_s not in properties_and_types.keys():
                            properties_and_types[str_s] =[]
                        p_type, p_value = self.get_property_type(str_o)
                        if (str_s,p_type, p_value) not in properties_and_types[str_s]:
                            properties_and_types[str_s].append((self.get_type(str_p), p_type, p_value))
                        triples.append((str_s,str_p,str_o))
            else:
                if str_s not in entities_and_type.keys():
                    entities_and_type[str_s] =[]
                triples.append((str_s,str_p,str_o))
                split_o = str_o.split('/')
                entities_and_type[str_s].append(split_o[len(split_o)-1])

#               properties_and_type[str(s)] = self.get_property_type(str(o))

        for e in entities_and_type:
            entities_and_type[e].sort()

        properties_and_types = self.get_classes_types(properties_and_types)

        return entities_and_type, triples, properties_and_types
