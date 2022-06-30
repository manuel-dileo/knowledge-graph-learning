import csv
from email import header
import os
from matplotlib.pyplot import cla
import matplotlib.pyplot as plt
import networkx as nx
import configparser 
from IPython.display import Image
from rdflib import RDF, Graph as RDFGraph, Literal, URIRef
from collections import defaultdict
import rdflib
import json
from networkx.readwrite import json_graph

from src.data.graph_modelling.functions import graph_to_json

class SemanticModelClass():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self.classes = {}
        self.leafs = {}
        self.triples = []
        self.closure_classes = {}
        self.closure_graph = nx.MultiDiGraph()
        self.ontology = rdflib.Graph()
        self.ontology.parse(self.config['ontology']['path'])

    def draw_result(self,graph, filename):
        node_label = nx.get_node_attributes(graph,'id')
        pos = nx.spring_layout(graph)
        p=nx.drawing.nx_pydot.to_pydot(graph)
        p.write_png(filename+'.png')
        Image(filename=filename+'.png')

    def draw_edge_labels(self, graph, filename):
        pos = nx.spring_layout(graph)
        nx.draw(
            graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in graph.nodes()}
        )
        labels = dict([((n1, n2), f'{n1}->{n2}')
                    for n1, n2, n3 in graph.edges])
        p = nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=labels,
            font_color='red'
        )
        plt.show()

    def parse(self):
        semantic_model = nx.MultiDiGraph()
        config_path=str(os.path.dirname(os.path.abspath(__file__))).split(os.sep)
        config_path = "/".join(config_path[0:len(config_path)-3])
        with open(config_path+self.config["semantic_model_csv"]["csv_path"], 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # skip the headers
            line_count = 0

            for row in csv_reader:
                if any(row): #to avoid empty lines
                    if row[1].strip() == "":
                        continue
                    
                    node_name = self.config["prefix"]["classes"] + row[1].strip()
                    self.classes[node_name]= self.classes.get(node_name, 0)

                    #if the element is an identifier
                    if row[3].strip().lower() == 'yes':
                        self.classes[node_name]= self.classes.get(node_name, 0)+int(row[4])
                        self.triples.append((node_name +str(self.classes[node_name]), row[2], row[0]))
                        semantic_model.add_edge(node_name +str(self.classes[node_name]),row[2], label = row[0])

                        #print(node_name +str(self.classes[node_name]))
                    
                    if row[2].strip() == "":
                        continue
                    else:
                        self.leafs[node_name + row[4].strip()] = self.config["prefix"]["properties"] + row[2].strip()
        return semantic_model

    def get_classes(self):
        return self.classes
    
    def get_leafs(self):
        return self.leafs

    def get_closure_classes(self):
        results = {}
        list_classes = list(self.classes.keys())
        superclasses = self.get_superclasses(list_classes)

        for class_name in list_classes:
            
            query = " SELECT DISTINCT ?closures WHERE {"+\
            "{ ?property rdfs:domain <"+ class_name+">."+\
            " ?property rdfs:range ?closures . ?closures a " + self.config["ontology"]["class"]+".}"+\
            " UNION { ?property rdfs:domain ?closures . ?property rdfs:range "+\
            " <"+class_name+">.  ?closures a " + self.config["ontology"]["class"]+".} }"
            
            result = self.ontology.query(query)

            if class_name + "0" not in results.keys():
                results[class_name + "0"] = 1
                #results.append(class_name + "0")

            for r in result:
                c_name = str(r.asdict()['closures'].toPython())
                if c_name+"0" not in results.keys():
                    if c_name in superclasses:
                        results[c_name + "0"] = 10
                    else:
                        results[c_name + "0"] = 1

                    #results.append(c_name+"0")

            num_classes = self.classes[class_name]
            if num_classes > 0:
                for i in range(1,num_classes+1):
                    if class_name+str(i) not in results.keys():
                        if class_name in superclasses:
                            results[class_name + str(i)] = 10
                        else:
                            results[class_name + str(i)] = 1

                        #results.append(class_name + str(i))

                    #for r in result:
                    #    c_name = str(r.asdict()['closures'].toPython())
                    #    while self.classes[class_name] > -1:
                    #        results.append(c_name+str(i))     
        self.closure_classes = results   
        return results

    def get_superclasses(self, classes):
        results = {}
        list_result = []
        for class_node in classes:
            query = "SELECT ?all_super_classes WHERE { <" +class_node +"> "+\
            " <"+self.config["prefix"]["subclass"]+ "> ?all_super_classes . }"

            result = self.ontology.query(query)
            for r in result:
                c_name = str(r.asdict()['all_super_classes'].toPython())
                if class_node in results.keys():
                    results[class_node].append(c_name)
                else:
                    results[class_node] = [c_name]
                list_result.append(c_name)

        return list_result


    def get_subclasses(self, class_node):
        query = "SELECT ?all_super_classes WHERE { ?all_super_classes "+\
        self.config["prefix"]["subclass"]+ " <"+class_node +">"
        result = self.ontology.query(query)
        return result
    

    def get_superclass_of(self, node):
        query = "SELECT ?super_class WHERE { <"+node +"> <"+\
        self.config["prefix"]["subclass"]+ "> ?super_class .}"
        result = self.ontology.query(query)
        res = []
        for r in result:
            res.append(str(r[0]))
        return res

    def is_subclass(self, candidate, superclass):
        superclass = superclass[0: len(superclass)-1]
        query = " SELECT ?subclass  WHERE { ?subclass <"+ self.config["prefix"]["subclass"] + "> <" +superclass +">. }"
        result = self.ontology.query(query)
        for r in result:
            if str(r) == candidate[0: len(candidate)-1]:
                return True
        return False

    def get_edges(self):
        for subj in self.closure_classes.keys():
            for obj in self.closure_classes.keys():
                query = " SELECT ?direct_properties WHERE {"+\
                " ?direct_properties <"+ self.config["prefix"]["domain"]+ "> <" +subj[:-1] +\
                ">. ?direct_properties <"+ self.config["prefix"]["range"]+ "> <" +obj[:-1]+"> .}"
  
                result = self.ontology.query(query)

                for r in result:
                    p_name = str(r.asdict()['direct_properties'].toPython())
                    weight = max(self.closure_classes[subj], self.closure_classes[obj])
                    #label = "w:" + str(weight) + " - "+str(p_name)
                    #self.closure_graph.add_edge(subj,obj, label)
                    self.closure_graph.add_edge(subj,obj, label = str(p_name), weight = weight)
                    #self.closure_graph.add_edge(subj,obj)

        return self.closure_graph

    def get_graph_closure(self):
         self.get_closure_classes()
         return self.get_edges()

    def get_relation_type(self,relation):
        r_split = relation.split("/")
        return r_split[len(r_split)-1]

    def update_graph_weights(self, closure, weights):

        for edge in closure.edges:
            u = edge[0]
            v = edge[1]
            rel = closure.get_edge_data(u,v)[0]
            u_type = self.get_relation_type(str(u)[:-1])
            v_type = self.get_relation_type(v)[:-1]
            rel_type = self.get_relation_type(rel['label'])
            try:
                rgcn_weight = weights[(u_type,rel_type,v_type)]
            except KeyError:
                rgcn_weight = 100
            rel["weight"] = abs(1-rgcn_weight*rel["weight"])

        return closure

    def set_graph_weights(self, closure, weights):
        for edge in closure.edges:
            u = edge[0]
            v = edge[1]
            rel = closure.get_edge_data(u,v)[0]
            u_type = self.get_relation_type(str(u)[:-1])
            v_type = self.get_relation_type(v)[:-1]
            rel_type = self.get_relation_type(rel['label'])

            try:
                rgcn_weight = weights[(u_type,rel_type,v_type)]
            except KeyError:
                rgcn_weight  = 100.0
            rel["weight"] = abs(rgcn_weight*1)
        return closure

    def graph_to_json(self,graph):
        data1 = json_graph.node_link_data(graph)
        s2 = json.dumps(
            data1
        )
        return s2

    def compute_closure_graph(self,semantic_model):
        closure_graph = nx.MultiDiGraph()
        superclass_subclass = {}

        tot_classes = []
        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                closure_graph.add_node(node)
                superclasses = self.get_superclass_of(node[0:len(node)-1])
                tot_classes.append(node)
                for superclass in superclasses:
                    superclass = superclass+"0"
                    tot_classes.append(superclass)
                    if superclass not in list(superclass_subclass.keys()):
                        superclass_subclass[superclass] = []
                    superclass_subclass[superclass].append(node) 

        for node in tot_classes:
            node = str(node)
            query = " SELECT DISTINCT ?property ?class WHERE {"+\
            " ?property rdfs:domain <"+ node[0:len(node)-1]+">."+\
            " ?property rdfs:range ?class . ?class a " + self.config["ontology"]["class"]+".}"
            
            result = self.ontology.query(query)
            
            for r in result:

                rel = str(r[0])
                obj = str(r[1])
                if obj not in tot_classes:
                    obj += "0"

                if node in list(superclass_subclass.keys()) and obj in list(superclass_subclass.keys()):
                    for n in superclass_subclass[node]:
                        for n2 in superclass_subclass[obj]:
                            closure_graph.add_edge(n, n2, label = rel, weight = 10)
                elif node in list(superclass_subclass.keys()):
                    for n in superclass_subclass[node]:
                        closure_graph.add_edge(n, obj, label = rel, weight = 10)
                elif obj in list(superclass_subclass.keys()):
                    for n2 in superclass_subclass[obj]:
                        closure_graph.add_edge(node, n2, label = rel, weight = 10)
                else:
                    closure_graph.add_edge(node, obj, label = rel, weight = 1)
        return closure_graph

    def algorithm(self, semantic_model):
        closure = self.compute_closure_graph(semantic_model)
        print(graph_to_json(closure))
        Uc = [] 
        Ut = []
        Et = []
        Er = []
        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                Uc.append(node)
            else:
                Ut.append(node)

        for edge in semantic_model.edges:
            label = semantic_model.get_edge_data(edge[0], edge[1])[0]
            if edge[0][0:4].startswith("http") and edge[1][0:4].startswith("http"):
                Er.append(label["label"])
            else:
                
                Et.append(label["label"])


        '''
        for uc in Uc:
            for edge in closure.edges:
                C1 = edge[0]
                if self.is_subclass(uc, C1):
                    us = uc


         def compute_closure_graph(self,semantic_model):
        closure_graph = nx.MultiDiGraph()

        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                closure_graph.add_node(node)
                superclasses = self.get_superclass_of(node[0:len(node)-1])

                for superclass in superclasses:
                    closure_graph.add_edge(node, str(superclass)+"0", label = 'subclass')

        triples_to_add = []
        for node in closure_graph.nodes:
            node = str(node)
            query = " SELECT DISTINCT ?property ?class WHERE {"+\
            " ?property rdfs:domain <"+ node[0:len(node)-1]+">."+\
            " ?property rdfs:range ?class . ?class a " + self.config["ontology"]["class"]+".}"
            
            result = self.ontology.query(query)

            for r in result:
                rel = str(r[0])
                obj = str(r[1]) + "0"
                triples_to_add.append((node, obj, rel))
            
        for triple in triples_to_add:
            closure_graph.add_edge(triple[0], triple[1], label = triple[2])

        return closure_graph
        '''