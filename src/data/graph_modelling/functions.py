from bs4 import ResultSet
from matplotlib.pyplot import draw
import networkx as nx
from networkx import Graph as NXGraphdef 
from networkx.readwrite import json_graph
import configparser
import os 
from rdflib import Graph as RDFGraph, URIRef
from IPython.display import Image
from collections import defaultdict
import rdflib
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import json

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
closure = nx.DiGraph()

def get_classes(graph):
    query = """
        SELECT DISTINCT ?class WHERE 
        {
            ?class a owl:Class. 
        }
    """
    result = graph.query(query)
    results = []
    for s in result:
        results.append(str(s))
    return results

def draw_result(graph, filename):
    node_label = nx.get_node_attributes(graph,'id')
    pos = nx.spring_layout(graph)
    node_label = nx.get_node_attributes(graph,'id')
    pos = nx.spring_layout(graph)
    p=nx.drawing.nx_pydot.to_pydot(graph)
    p.write_png(filename+'.png')
    Image(filename=filename+'.png')

def graph_to_json(graph):
    data1 = json_graph.node_link_data(graph)
    s2 = json.dumps(
        data1
    )
    return s2

'''
def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def get_neighbours(graph, node):
    if graph.has_node(node):
        graph.in_edges(node)
        graph.out_edges(node) 

def get_closure(semantic_types, ontology):
    for node in semantic_types:
        print(node)
        neighbours = get_neighbours(ontology, node)
        print(neighbours)
'''

