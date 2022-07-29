from asyncio import proactor_events
from tkinter import S
import rdflib
from rdflib import URIRef, RDF
from rdflib.namespace import Namespace
import configparser
import os
import src.data.utils as utils


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
        self.entities_and_type = {}
        self.properties_and_type ={}

    def set_entities_and_type(self, graph, use_properties = False):
        relations = []
        triples = []
        # Process the Knowledge Graph

        for s, p, o in graph:
            str_s = str(s)
            str_p = str(p)
            str_o = str(o)
            if str_p != str(RDF.type):
                if not str_s in self.entities_and_type.keys():
                    self.entities_and_type[(str_s)] =[]
                if not str_p in relations:
                    relations.append(str_p)

                if str_o.find('^^') == -1:
                    if not str_o in self.entities_and_type.keys():
                        self.entities_and_type[str_o]=[]
                    triples.append((str_s,str_p,str_o))
                else:
                    if use_properties:
                        if str_s not in self.properties_and_type.keys():
                            self.properties_and_type[str_s] =[]
                        p_type, p_value = utils.get_property_type(str_o)
                        if (str_s,p_type, p_value) not in self.properties_and_type[str_s]:
                            self.properties_and_type[str_s].append((utils.get_type(str_p), p_type, p_value))
                        triples.append((str_s,str_p,str_o))
            else:
                if str_s not in self.entities_and_type.keys():
                    self.entities_and_type[str_s] =[]
                triples.append((str_s,str_p,str_o))
                split_o = str_o.split('/')
                self.entities_and_type[str_s].append(split_o[len(split_o)-1])

#               properties_and_type[str(s)] = self.get_property_type(str(o))

        for e in self.entities_and_type:
            self.entities_and_type[e].sort()

        return triples



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
    '''
    def get_class_from_property(self):
        Q = " SELECT ?property ?class WHERE {?property rdfs:domain ?class; rdf:type owl:DatatypeProperty. }"
        results = {}
        result = self.ontology.query(Q)
        for res in result:
            results[self.get_type(str(res[0]))]= self.get_type(str(res[1]))
        return results
    '''
    def get_classes_types(self):
        new_properties_and_types = {}
        for s in list(self.properties_and_type.keys()):
            for element in self.properties_and_type[s]:
                s_class = self.entities_and_type[s]
                
                if s not in new_properties_and_types:
                    new_properties_and_types[s] = []
                new_properties_and_types[s].append((s_class[0], element[0], element[1], element[2]))
        
        self.properties_and_type = new_properties_and_types

    def disambiguate_multiple_types(self, s,p,o): 
        for subtype_subj in self.entities_and_type[str(s)]:
            if len(self.entities_and_type[str(o)]) > 1:
                for subtype_obj in self.entities_and_type[str(o)]:
                    possible_rels = self.get_possible_types( subtype_subj, subtype_obj)

                    if len(possible_rels) == 0:
                        continue   
                    
                    p = utils.get_type(p)
                    for rel in possible_rels:
                        if utils.get_type(rel) == p:
                            return (subtype_subj,subtype_obj)
            else:
                subtype_obj = self.entities_and_type[str(o)][0]
                possible_rels = self.get_possible_types(subtype_subj, subtype_obj)
                if len(possible_rels) == 0:
                        continue
                p = utils.get_type(p)   
                for rel in possible_rels:
                    if utils.get_type(rel) == p:
                        return (subtype_subj,  subtype_obj)
            
        return ("","")      

    def get_count(self):
        self.get_classes_types()
        entity_types_count = {}
        property_types_count = {}

        for entity in self.entities_and_type.keys():
            tipo = self.entities_and_type[entity][0]
            if tipo != "":
                entity_types_count[tipo] = entity_types_count.get(tipo, 0)+1

        for subj in self.properties_and_type.keys():
            for class_name, prop_name, prop_type, prop_value in self.properties_and_type[subj]:
                key = (class_name, subj, prop_name, prop_type)
                property_types_count[key] = property_types_count.get(key, 0)+1
        return entity_types_count, property_types_count

    def clean_triples(self, triples):
        new_triples = []
        added_types = []

        for s,p,o in triples:
            str_s = str(s)
            str_p = str(p)
            str_o = str(o)

            if str_p != str(RDF.type):
            #if s1 in list(properties_and_types.keys()):
                if str_o.find("^^") != -1:
                    #x = properties_and_type[str(s)]
                    new_triples.append((s, p, o))
                if str_s in list(self.entities_and_type.keys()) and str_o in list(self.entities_and_type.keys()):
                    #se è una relazione tra classi

                    #se il soggetto o l'oggetto ha più di un tipo 
                    if len(self.entities_and_type[str_s]) > 1 or len(self.entities_and_type[str_o]) > 1:
                        new_subj_type, new_obj_type = self.disambiguate_multiple_types(s,p,o)
                        if((new_subj_type, new_obj_type) == ("","") 
                            or new_subj_type == "" 
                            or new_obj_type ==""):
                            continue

                
                        self.entities_and_type[str_s] = [new_subj_type]
                        self.entities_and_type[str_o] = [new_obj_type]

                        if s not in added_types:
                            new_triples.append((s, 
                                                str(RDF.type),
                                                self.config["prefixes"]["ontology"]+ new_subj_type
                                                ))

                            added_types.append(s)

                        if o not in added_types:
                            new_triples.append((o,
                                                str(RDF.type),
                                                self.config["prefixes"]["ontology"]+ new_obj_type 
                                                ))
                            added_types.append(o)
                    else: 
                        if s not in added_types:
                            new_triples.append((s, 
                                                str(RDF.type),
                                                self.config["prefixes"]["ontology"]+self.entities_and_type[str_s][0] ))
                            added_types.append(s)
                        if o not in added_types and str_o.find("^^") == -1:
                            new_triples.append((o, 
                                                str(RDF.type),
                                                self.config["prefixes"]["ontology"]+self.entities_and_type[str_o][0] ))
                            added_types.append(o)
                    if(s,p,o) not in new_triples:
                        new_triples.append((s, p, o))
            else:
                if s not in added_types and len(self.entities_and_type[str_s]) == 1: 
                #controllo solo s perché la relazione p indica che o è il tipo, 
                # verifico che non ci sia piu di un tipo altrimenti rimando l'aggiunta a quando viene
                #disambiguato
                    new_triples.append((s, p, o))
                    added_types.append(s)

        return new_triples

   
    def get_subject_object(self, triples, entity_types_count, property_types_count = {}):
        subject_dict = {}
        object_dict = {}

        index_dict = {t:{'count': 0} for t in entity_types_count.keys()}

        for class_name, subject,rel, p_type in property_types_count.keys():
            index_dict[p_type] = {'count':0}
            if class_name not in index_dict.keys():
                index_dict[class_name] = {'count':0}  
                
        triples.sort()
        for triple in triples:
            s = str(triple[0])
            p = str(triple[1])
            o = str(triple[2])
            type_triples = []
            
            if p != str(RDF.type):
                s_type = self.entities_and_type[s][0] 
                p_type = utils.get_type(p)

                if o.find("^^") == -1:
                    o_type = self.entities_and_type[o][0]
                else: 
                    o_type = utils.get_property_type(o)[0]
                
                type_triples.append((s_type,p_type, o_type))

                for s_type,p_type,o_type in type_triples:
                    if(s_type != "" and o_type != ""):
                        key_t = (s_type, p_type, o_type)
                        
                        if key_t not in subject_dict.keys():
                            subject_dict[key_t] = []
                            object_dict[key_t] = []
                            
                        if s not in index_dict[s_type]:
                            index_dict[s_type][s] = index_dict[s_type]['count']
                            index_dict[s_type]['count'] = index_dict[s_type]['count']+1
                        s_index = index_dict[s_type][s]
                            
                        if o not in index_dict[o_type]:
                            index_dict[o_type][o] = index_dict[o_type]['count']
                            index_dict[o_type]['count'] = index_dict[o_type]['count']+1
                        o_index = index_dict[o_type][o]
                            
                        subject_dict[key_t].append(s_index)
                        object_dict[key_t].append(o_index)
                
                #data[s_type, p_type, o_type].edge_index[0].append(entities.index(str(s)))
                #data[s_type, p_type, o_type].edge_index[1].append(entities.index(str(o)))
        return subject_dict, object_dict, self.properties_and_type

    
