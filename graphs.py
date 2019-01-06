
import os
from graphviz import Digraph

from Genome import *

# set path for graphviz executables
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def create_graph(gene, generation, gene_number):
    """
    creates a digraph for the given neural network and stores it in the given gen folder
    :param gene:  a neural network gene
    :param generation: current generation
    :param gene_number: gene number in the population
    :return: none
    """
    f = Digraph('finite_state_machine', filename='saved_data/gen'+str(generation)+'/'+str(gene_number))
    f.attr(rankdir='BT', size='8,5')

    f.attr('node', shape='circle')
    list_of_connections = gene.return_connections()
    list_of_nodes = gene.return_nodes()

    with f.subgraph(name='cluster_input') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for node in list_of_nodes:
            if node[1] == 'input':
                c.node(str(node[0]))
        c.attr(label='input')

    with f.subgraph(name='cluster_output') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for node in list_of_nodes:
            if node[1] == 'output':
                c.node(str(node[0]))
        c.attr(label='output')
    for connection in list_of_connections:
        if connection[3]:
            f.edge(str(connection[0]), str(connection[1]), str("%.2f" % round(connection[2], 2)))
        else :
            f.edge(str(connection[0]), str(connection[1]), str("%.2f" % round(connection[2], 2)), style="dotted")

    f.render()

def create_graph_best(gene, generation,species_number):
    f = Digraph('finite_state_machine', filename='saved_data/gen'+str(generation)+'/species'+str(species_number))
    f.attr(rankdir='BT', size='8,5')

    f.attr('node', shape='circle')
    list_of_connections = gene.return_connections()
    list_of_nodes = gene.return_nodes()

    with f.subgraph(name='cluster_input') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for node in list_of_nodes:
            if node[1] == 'input':
                c.node(str(node[0]))
        c.attr(label='input')

    with f.subgraph(name='cluster_output') as c:
        c.attr(style='filled', color='lightgrey')
        c.node_attr.update(style='filled', color='white')
        for node in list_of_nodes:
            if node[1] == 'output':
                c.node(str(node[0]))
        c.attr(label='output')
    for connection in list_of_connections:
        if connection[3]:
            f.edge(str(connection[0]), str(connection[1]), str("%.2f" % round(connection[2], 2)))
        else :
            f.edge(str(connection[0]), str(connection[1]), str("%.2f" % round(connection[2], 2)), style="dotted")

    f.render()