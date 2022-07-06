import csv
from email import header
import os
from re import S, sub
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
#        self.ontology.parse(self.config['ontology']['path'])
        self.ontology.parse("/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/external/ontologia_ereditarieta_livelli.ttl")
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
                    #self.classes[node_name]= self.classes.get(node_name, 0)

                    #if the element is an identifier
                    if row[3].strip().lower() == 'yes':
                        self.classes[node_name]= self.classes.get(node_name, -1)+1
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
        subclasses =[]
        query = "SELECT ?all_super_classes WHERE { ?all_super_classes rdfs:subClassOf "+\
                "<"+class_node +">.}"
        result = self.ontology.query(query)
        for r in result:
            subclasses.append(str(r[0]))
        return subclasses
    

    def get_superclass(self, node):
        query = "SELECT ?super_class WHERE { <"+node +"> <"+\
        self.config["prefix"]["subclass"]+ "> ?super_class .}"
        result = self.ontology.query(query)
        res = []
        for r in result:
            res.append(str(r[0]))
        return res

    def get_outgoing_links(self, node):
        query = "SELECT ?rel WHERE { ?rel rdfs:domain <"+node +"> .}"
        result = self.ontology.query(query)
        res = []
        for r in result:
            res.append(str(r[0]))
        return res

    def get_out_links_and_obj(self,node):
        query = "SELECT ?rel ?relatedClass WHERE { ?rel rdfs:domain <"+node +">; "+\
            "rdf:type owl:ObjectProperty; rdfs:range ?relatedClass .}"
        result = self.ontology.query(query)
        res = []
        for r in result:
            res.append((str(r[0]), str(r[1])))
        return res

    def get_ingoing_links_and_subj(self,node):
        query = "SELECT ?rel ?relatedClass WHERE { ?rel rdfs:range <"+node +">.; "+\
            "rdf:type owl:ObjectProperty; rdfs:range ?relatedClass .}"
        result = self.ontology.query(query)
        res = []
        for r in result:
            res.append((str(r[1]), str(r[0])))
        return res

    def is_subclass(self, candidate, superclass):
        superclass = superclass[0: len(superclass)-1]
        query = " SELECT ?subclass  WHERE { ?subclass rdfs:subClassOf <" +superclass +">. }"
        result = self.ontology.query(query)
        for r in result:
            if str(r[0]) == candidate:
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
        tot_instances = []
        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                closure_graph.add_node(node)
                superclasses = self.get_superclass(node[0:len(node)-1])
                tot_instances.append(node)
                tot_classes.append(node[0:len(node)-1])
                for superclass in superclasses:
                    tot_classes.append(superclass)
                    superclass = superclass+"0"
                    tot_instances.append(superclass)
                    if superclass not in list(superclass_subclass.keys()):
                        superclass_subclass[superclass] = []
                    superclass_subclass[superclass].append(node) 
        
        for node in tot_instances:
            node = str(node)[0:len(node)-1]
            query = " SELECT DISTINCT ?property ?class WHERE {"+\
            " ?property rdfs:domain <"+ node +">."+\
            " ?property rdfs:range ?class . ?class a " + self.config["ontology"]["class"]+".}"
            
            result = self.ontology.query(query)
            for r in result:
                rel = str(r[0])
                obj = str(r[1])

                node_istances = self.classes.get(node, 0)
                obj_istances = self.classes.get(obj, 0)

                for h in range(0, node_istances+1):
                    for h2 in range(0, obj_istances+1):
                        if node +str(h) in list(superclass_subclass.keys()) and obj+str(h2) in list(superclass_subclass.keys()):
                            weight = 10
                        elif node +str(h) in list(superclass_subclass.keys()) or obj+str(h2) in list(superclass_subclass.keys()):
                            weight = 10
                        else:
                            weight = 1
                        
                        if not self.exists_edge(closure_graph, node +str(h), obj+str(h2), rel):
                            closure_graph.add_edge(node +str(h), obj+str(h2), label = rel, weight = weight)
        return closure_graph

    def compute_closure_node(self,node):
        closure_node = nx.MultiDiGraph()
        superclass_subclass = {}

        tot_classes = []

        if node[0:4].startswith("http"): #non è una proprietà
            superclasses = self.get_superclass(node)
            tot_classes.append(node)
            for superclass in superclasses:
                tot_classes.append(superclass)

                if superclass not in list(superclass_subclass.keys()):
                    superclass_subclass[superclass] = []
                superclass_subclass[superclass].append(node) 
        
        for node in tot_classes:
            node = str(node)
            query = " SELECT DISTINCT ?property ?class WHERE {"+\
            " ?property rdfs:domain <"+ node +">."+\
            " ?property rdfs:range  ?class . ?class a " + self.config["ontology"]["class"]+".} "
            
            result = self.ontology.query(query)
            for r in result:
                rel = str(r[0])
                obj = str(r[1])

                if node in list(superclass_subclass.keys()) and obj in list(superclass_subclass.keys()):
                    weight = 10
                elif node in list(superclass_subclass.keys()) or obj in list(superclass_subclass.keys()):
                    weight = 10
                else:
                    weight = 1
                
                if not self.exists_edge(closure_node, node, obj, rel):
                    closure_node.add_edge(node, obj, label = rel, weight = weight)

            query = " SELECT DISTINCT ?property ?class WHERE {"+\
            " ?property rdfs:range <"+ node +">."+\
            " ?property rdfs:domain  ?class . ?class a " + self.config["ontology"]["class"]+".} "
            
            result = self.ontology.query(query)
            for r in result:
                rel = str(r[0])
                subj = str(r[1])

                if node in list(superclass_subclass.keys()) and subj in list(superclass_subclass.keys()):
                    weight = 10
                elif node in list(superclass_subclass.keys()) or subj in list(superclass_subclass.keys()):
                    weight = 10
                else:
                    weight = 1
                
                if not self.exists_edge(closure_node, subj, node, rel):
                    closure_node.add_edge(subj, node, label = rel, weight = weight)
        return closure_node


    def exists_edge(self,graph, u, v, label):
        edges = graph.get_edge_data(u, v)

        if edges == None:
            return False

        for i in range(0, len(edges)):
            if label == edges[i]["label"]:
                return True
        return False


    def get_distance(self, C1, C2):
        if C1 == C2:
            return 0
        superclass = self.get_superclasses([C2])
        if len(superclass) == 0:
            return 0
        if C1 in superclass:
            return 1
        
        return 1+ self.get_distance(C1, superclass[0])

    def get_distance_undirected(self, C1,C2):
        n = self.get_distance(C1, C2)
        m = self.get_distance(C2, C1)
        return max(n,m)

    def homogenize_lists(self, us_list, ut_list):
        if len(us_list) > len(ut_list):
            for i in range(len(ut_list), len(us_list)):
                ut_list.append(ut_list[len(ut_list)-1])
        if len(us_list) < len(ut_list):
            for i in range(len(us_list), len(ut_list)):
                us_list.append(us_list[len(us_list)-1])
        return (us_list, ut_list)


    def algorithm(self,semantic_model):
        #closure = self.compute_closure_node("http://dbpedia.org/ontology/Director")
        #return closure
        Uc_occurrences = {}

        Uc = [] 
        Ut = []
        Et = []
        Er = []
        Uc_ini = []
        #init UC and Ut
        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                Uc.append(node)
                Uc_ini.append(node)
                Uc_occurrences[node[0:len(node)-1]] = Uc_occurrences.get(node[0:len(node)-1],0)+1
            else:
                Ut.append(node)

        #Init Et and Er
        for edge in semantic_model.edges:
            label = semantic_model.get_edge_data(edge[0], edge[1])[0]
            if edge[0][0:4].startswith("http") and edge[1][0:4].startswith("http"):
                Er.append(label["label"])
            else:
                
                Et.append(label["label"])

                #print(edge[0],edge[1], closure_C.get_edge_data(edge[0],edge[1]))
        closure_classes = []
        
        for uc in Uc_ini:
            C = uc[0: len(uc)-1]
            closure_classes.append(C)
            closure_C = self.compute_closure_node(C)
            for edge in closure_C.out_edges:
                if edge[0] not in closure_classes:
                    closure_classes.append(edge[0])
                if edge[0] not in closure_classes:
                    closure_classes.append(edge[1])

        for uc in Uc_ini:
            us = ""
            C = uc[0: len(uc)-1]

            closure_C = self.compute_closure_node(C)
            #self.draw_result(closure_C, "/home/sara/Desktop/fase2/git_repo/knowledge-graph-learning/data/graph_images/closure_node111")
            
            for edge in closure_C.out_edges:
                #print(edge[0],edge[1], closure_C.get_edge_data(edge[0],edge[1]))
                epsilon = 10
                C1 = edge[0]
                C2 = edge[1]
                relations=[]
                rel = closure_C.get_edge_data(C1,C2)
                for i in range(len(rel)):
                    relations.append(rel[i]["label"])

                us_list =[]
                ut_list =[]
                if self.is_subclass(C, C1) or C==C1:
                    us_list.append(uc)
                else:
                    uc1 = C1+"0"
                    if uc1 not in Uc:
                        if not self.is_superclass_or_subclass_of(uc1, Uc):
                            Uc.append(uc1)
                            if C1 not in closure_classes:
                                closure_classes.append(C1)
                            us_list.append(uc1)
                            if C1 not in Uc_occurrences:
                                Uc_occurrences[C1] = 1
                        else:
                            subclasses = self.get_subclasses(C1)
                            superclasses = self.get_superclass(C1)
                            if len(subclasses)!= 0:
                                for subclass in subclasses:
                                    k = Uc_occurrences.get(subclass,0)
                                    for i in range(k):
                                        us = subclass+str(i)
                                        if subclass in closure_classes:
                                            us_list.append(us)
                                    if k == 0 and subclass in closure_classes:
                                        us_list.append(subclass+"0")
                    else:
                        us_list.append(uc1)

                if self.is_subclass(C, C2) or C == C2:
                    ut_list.append(uc)
                else:
                    uc2 = C2+"0"
                    if uc2 not in Uc:
                        
                        if not self.is_superclass_or_subclass_of(uc2, Uc):
                            Uc.append(uc2)
                            if C2 not in closure_classes:
                                closure_classes.append(C2)
                            ut_list.append(uc2)
                            if C2 not in Uc_occurrences:
                                Uc_occurrences[C2] = 1
                            superclasses = self.get_superclass(C2)
                            if len(superclasses) != 0:
                                for superclass in superclasses:
                                    subclasses = self.get_subclasses(superclass)
                                    for subclass in subclasses:
                                        if subclass != C2 and subclass in closure_classes:
                                            ut_list.append(subclass+"0")
                        else:
                            subclasses = self.get_subclasses(C2)
                            if len(subclasses)!= 0:
                                for subclass in subclasses:
                                    k = Uc_occurrences.get(subclass,0)
                                    for i in range(k):
                                        ut = subclass+str(i)
                                        if subclass in closure_classes:
                                            ut_list.append(ut)
                                    if k == 0 and subclass in closure_classes:
                                        ut_list.append(subclass+"0")
                    else:
                        ut_list.append(uc2)

                us_list, ut_list = self.homogenize_lists(us_list, ut_list)

                for r in relations:
                    for i in range(len(us_list)):
                        us = us_list[i]
                        ut = ut_list[i]

                        H = Uc_occurrences.get(us[0:len(us)-1],0)
                        K = Uc_occurrences.get(ut[0:len(ut)-1])
                        h = int(us[len(us)-1:])
                        k = int(ut[len(ut)-1:])

                        Pr_source = self.get_distance_undirected(C1,us)
                        Pr_dest = self.get_distance_undirected(C2,ut)
                        Pr = (Pr_source + Pr_dest)*epsilon

                        if us != ut and (us, r, ut, Pr) not in Er and (ut, r, us, Pr) not in Er:
                            if (h == k) or (H <= K and h == H-1 and k > h) or (K-1 == k and h > k):
                                Er.append((us,r,ut, Pr))

                    '''
                    if len(us_list) < len(ut_list):
                        if len(us_list) == 0:
                                break
                        for i in range(n_min, n_max):
                            us = us_list[n_min-1]
                            ut = ut_list[i]
                            if( us != ut  and 
                                (us, r, ut) not in Er 
                                and (ut, r, us) not in Er):
                                Pr_source = self.get_distance_undirected(C1,us)
                                Pr_dest = self.get_distance_undirected(C2,ut)
                                Pr = (Pr_source + Pr_dest)*epsilon
                                Er.append((us,r,ut,Pr)) 
                    
                    elif len(us_list) > len(ut_list):
                        if len(ut_list) == 0:
                            break
                        for i in range(n_min, n_max):

                            Pr_source = self.get_distance_undirected(C1,us)
                            Pr_dest = self.get_distance_undirected(C2,ut)
                            Pr = (Pr_source + Pr_dest)*epsilon
                            us = us_list[i]
                            ut = ut_list[n_min-1]
                            if( us != ut and 
                                (us, r, ut) not in Er 
                                and (us, r, ut) not in Er):
                                Er.append((us,r,ut, Pr))                 
                    '''
        return (Uc, Er)


    def is_superclass_or_subclass_of(self, uc, Uc):
        for uq in Uc:
            if self.is_subclass(uq[0:len(uq)-1], uc):
                return True

        for uq in Uc:
            if self.is_subclass(uc[0:len(uc)-1], uq):
                return True
        
        return False

    def graph_creation_algorithm(self,semantic_model):
        #closure = self.compute_closure_node("http://dbpedia.org/ontology/Director")
        #return closure
        Uc_occurrences = {}
        graph = nx.MultiDiGraph()
        graph_ini = nx.MultiDiGraph()

        #init UC and Ut
        for node in semantic_model.nodes:
            if node[0:4].startswith("http"):
                graph.add_node(node)
                graph_ini.add_node(node)
                Uc_occurrences[node[0:len(node)-1]] = Uc_occurrences.get(node[0:len(node)-1],0)+1

        #Init Et and Er
        for edge in semantic_model.edges:
            label = semantic_model.get_edge_data(edge[0], edge[1])[0]
            if edge[0][0:4].startswith("http") and edge[1][0:4].startswith("http"):
                graph.add_edge(edge[0],edge[1])
                graph_ini.add_edge(edge[0],edge[1])

        for uc in graph_ini.nodes:
            us = ""
            h = int(uc[len(uc)-1:])
            C = uc[0: len(uc)-1]

            closure_C = self.compute_closure_node(C)
            for edge in closure_C.out_edges:
                C1 = edge[0]
                C2 = edge[1]
                relations=[]

                rel = closure_C.get_edge_data(C1,C2)
                for i in range(len(rel)):
                    relations.append(rel[i]["label"])

                us_list =[]
                ut_list =[]
                if self.is_subclass(C, C1) or C==C1:
                    us_list.append(uc)
                else:
                    uc1 = C1+"0"
                    if uc1 not in graph:
                        if not self.is_superclass_or_subclass_of(uc1, graph.nodes):
                            graph.add_node(uc1)
                            us_list.append(uc1)
                            if C1 not in Uc_occurrences:
                                Uc_occurrences[C1] = 1
                        else:
                            subclasses = self.get_subclasses(C1)
                            superclasses = self.get_superclass(C1)
                            if len(subclasses)!= 0:
                                for subclass in subclasses:
                                    k = Uc_occurrences.get(subclass,0)
                                    for i in range(k):
                                        us = subclass+str(i)
                                        if graph.has_node(us):
                                            us_list.append(us)
                    else:
                        us_list.append(uc1)

                if self.is_subclass(C, C2) or C == C2:
                    ut_list.append(uc)
                else:
                    uc2 = C2+"0"
                    if uc2 not in graph:
                        if not self.is_superclass_or_subclass_of(uc2, graph.nodes):
                            graph.add_node(uc2)
                            ut_list.append(uc2)
                            if C2 not in Uc_occurrences:
                                Uc_occurrences[C2] = 1
                        else:
                            subclasses = self.get_subclasses(C2)
                            if len(subclasses)!= 0:
                                for subclass in subclasses:
                                    k = Uc_occurrences.get(subclass,0)
                                    for i in range(k):
                                        ut = subclass+str(i)
                                        if graph.has_node(ut):
                                            ut_list.append(ut)
                    else:
                        ut_list.append(uc2)

                n_min = min(len(us_list), len(ut_list))
                n_max = max(len(us_list), len(ut_list))
                
                for r in relations:
                    for i in range(n_min):
                        ut = ut_list[i]
                        us = us_list[i]
                        if us != ut and not self.exists_edge(graph, us, ut, r) and not self.exists_edge(graph, ut, us, r):
                            graph.add_edge(us,ut,label = r)
                    
                    if len(us_list) < len(ut_list):
                        for i in range(n_min+1, n_max):
                            us = us_list[n_min]
                            ut = ut_list[i]
                            if( us != ut  and 
                                not self.exists_edge(graph, us, ut, r) 
                                and not self.exists_edge(graph, ut, us, r)):
                                graph.add_edge(us,ut,label = r)
                    elif len(us_list) > len(ut_list):
                        for i in range(n_min+1, n_max):
                            ut = ut_list[n_min]
                            us = us_list[i]
                            if( us != ut and 
                                not self.exists_edge(graph, us, ut, r) 
                                and not self.exists_edge(graph, ut, us, r)):
                                graph.add_edge(us,ut,label = r)

        return graph