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

    def get_relation_type(self, relation):
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

    def disambiguate_multiple_types(self, entities_and_type, s,p,o): 
        
        for subtype_subj in entities_and_type[str(s)]:

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

    def get_count(self, entities_and_type):
        entity_types_count = {}
        entities = []
        for entity in entities_and_type.keys():
            tipo = entities_and_type[entity][0]
            if tipo != "":
                entity_types_count[tipo] = entity_types_count.get(tipo, 0)+1
                entities.append(entity)
        return entity_types_count, entities

    def clean_triples(self, triples, entities_and_type):
        new_triples = []
        added_types = []

        for s,p,o in triples:
            if len(entities_and_type[str(s)]) > 1:
                new_subj_type, new_obj_type = self.disambiguate_multiple_types(entities_and_type,s,p,o)
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
            else:  
                if s not in added_types:
                    new_triples.append((s, str(RDF.type),self.config["prefixes"]["ontology"]+entities_and_type[str(s)][0] ))
                    added_types.append(s)
                if o not in added_types:
                    new_triples.append((o, str(RDF.type),self.config["prefixes"]["ontology"]+entities_and_type[str(o)][0] ))
                    added_types.append(o)
                new_triples.append((s, p, o))
        
        return new_triples


    def get_subject_object(self, entities, triples, entities_and_type, entity_types_count):
        subject_dict = {}
        object_dict = {}

        new_triples= self.clean_triples(triples, entities_and_type)

        index_dict = {t:{'count': 0} for t in entity_types_count.keys()}
        new_triples.sort()
        for triple in new_triples:
            s = str(triple[0])
            p = str(triple[1])
            o = str(triple[2])

            if s in entities and o in entities:
                p_type = self.get_relation_type(p)
                s_type = entities_and_type[s][0]
                o_type = entities_and_type[o][0]

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

    def set_entities_and_type(self):
        entities_and_type = {}
        relations = []
        triples = []
        triple_properties=[]
        # Process the Knowledge Graph
        g = rdflib.Graph()
        g.parse(self.config["kg"]["path"], format=self.config["kg"]["format"])
        for s, p, o in g:
            if str(p) != str(RDF.type):
                if not str(s) in entities_and_type.keys():
                    entities_and_type[(str(s))] =[]
                if not str(p) in relations:
                    relations.append(str(p))

                if str(o).find('^^') == -1:
                    if not str(o) in entities_and_type.keys():
                        entities_and_type[str(o)]=[]
                    triples.append((s,p,o))
                else:
                    triple_properties.append((str(s),str(p),str(o)))
                
            else:
                if str(s) not in entities_and_type.keys():
                    entities_and_type[str(s)] =[]
                
                split_o = str(o).split('/')
                entities_and_type[str(s)].append(split_o[len(split_o)-1])
        for e in entities_and_type:
            entities_and_type[e].sort()

        return entities_and_type, triples
