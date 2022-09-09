import networkx as nx
from networkx.readwrite import json_graph
import json
import category_encoders as ce
import pandas as pd
from read_files_helper import *
from global_variables import *


def generate_datasets():
    transpose_error_and_syndrom_files()

    error_vectors, syndrom_vectors, H, H_orth, m1, m2, n, lv = read_files(data_dir)

    split_index = 4500

    datasetType = "train"
    generate_dataset(datasetType, error_vectors[:split_index], syndrom_vectors[:split_index], H, H_orth, m1, m2, n, lv)

    datasetType = "test"
    generate_dataset(datasetType, error_vectors[split_index:], syndrom_vectors[split_index:], H, H_orth, m1, m2, n, lv)

    """
    datasetType = "validation"
    generate_dataset(datasetType, error_vectors, syndrom_vectors)
    """


def generate_dataset(datasetType, error_vectors, syndrom_vectors, H, H_orth, m1, m2, n, lv):

    numGraphs = len(error_vectors)
    numVariableNodes = len(error_vectors[0]) #=2n
    numFactorNodes = len(syndrom_vectors[0]) #=m1 + m2
    if numGraphs == 0:
        return

    jsonFilePath = open_json_dataset_file(datasetType)
    encoded_variable_node_indices = encode_variable_node_indices(numVariableNodes)

    for iGraph in range(numGraphs):
        if iGraph % 10 == 0:
            print(iGraph)

        G = nx.DiGraph()  # directed graph
        #G = nx.Graph()  # undirected graph

        error_vector = error_vectors[iGraph]
        syndrom_vector = syndrom_vectors[iGraph]


        # posto je u ovom primjeru lv konstantan necemo na uzivati za inpute
        # TODO kada ga bude bilo potrebno uvaziti, mora vidjeti da li da to bude input u variable nodes ili neki master node koji je
        # povezan sa svim nodeovima (ili svim varijablama)
        add_variable_nodes(G, error_vector, encoded_variable_node_indices, numVariableNodes, iGraph)

        add_factor_nodes(G, syndrom_vector, numFactorNodes, numVariableNodes)

        if G.number_of_nodes() != numFactorNodes + numVariableNodes:
            print("ERROR: G.number_of_nodes() != numFactorNodes + numVariableNodes")
            return

        add_graph_edges(G, H, numFactorNodes, numVariableNodes)

        #connect_nodes_to_second_neighbours(G, numVariableNodes, should_connect_node_to_second_neighbours=True)

        #print("Graph diameter is: ", nx.diameter(G))
        #return


        # Ovaj kod ispod vazi samo za undirected graphs, tako da da bi radio, moras zakomentarisati ove linije:
        #G = nx.DiGraph()  # directed graph
        #G.add_edge(str(iVariable), str(factorNodeIndex))
        # a odkomentarisati ovu:
        #G = nx.Graph()  # undirected graph
        # uglavnom za kod 50 kaze da su dijametri podgrafova 8, a za kod 400 su dijametri 10!
        #graphs = (G.subgraph(c).copy() for c in nx.connected_components(G))
        #for gr in graphs:
            #print("Graph diameter is: ", nx.diameter(gr))
        #return

        parced_graph = json_graph.node_link_data(G)
        with open(jsonFilePath, 'a') as f:
            json.dump(parced_graph, f)
            f.write(",")

    with open(jsonFilePath, mode="r+") as file:
        file.seek(os.stat(jsonFilePath).st_size - 1)  # override the last comma in the file
        file.write("]")


def encode_variable_node_indices(numVariableNodes):
    data = pd.DataFrame(
        {'nodeIdx': [i for i in range(numVariableNodes)]})
    encoder = ce.BaseNEncoder(cols=['nodeIdx'], return_df=False, base=2)
    encoded_variable_node_indices = encoder.fit_transform(data)
    return encoded_variable_node_indices


def open_json_dataset_file(datasetType):
    if datasetType == "test":
        with open(os.path.join('data/test', 'data.json'), 'w') as json_file:
            json_file.write("[")
            jsonFilePath = os.path.join('data/test', 'data.json')
    elif datasetType == "validation":
        with open(os.path.join('data/validation', 'data.json'), 'w') as json_file:
            json_file.write("[")
            jsonFilePath = os.path.join('data/validation', 'data.json')
    else:
        jsonFilePath = os.path.join('data/train', 'data.json')
        with open(os.path.join('data/train', 'data.json'), 'w') as json_file:
            json_file.write("[")
    return jsonFilePath


def add_graph_edges(G, H, numFactorNodes, numVariableNodes):
    # numVariableNodes = len(error_vectors[0])  =2n
    # numFactorNodes = len(syndrom_vectors[0])  =2m
    # variable nodes: 0..numVariableNodes-1
    # factor nodes: numVariables..numVariableNodes + numFactorNodes - 1
    #G.add_edge(str(0), str(numFactorNodes + numVariableNodes - 1))
    #G.add_edge(str(numFactorNodes + numVariableNodes - 1), str(0))

    for iFactor in range(numFactorNodes):
        for iVariable in range(numVariableNodes):
            if abs(float(H[iFactor, iVariable])) > 0.0001:
                factorNodeIndex = iFactor + numVariableNodes
                G.add_edge(str(factorNodeIndex), str(iVariable))
                # IGNNITION received as input an undirected graph, even though it only
                # supports (at the moment) directed graphs -> therefore we must double the number of edges.
                G.add_edge(str(iVariable), str(factorNodeIndex))


def add_variable_nodes(G, error_vector, encoded_variable_node_indices, numVariableNodes, iGraph):
    for iVar in range(numVariableNodes):
        index_encoding = encoded_variable_node_indices[iVar].tolist()
        #G.add_node(str(iVar), entity='variableNode', iGraph=iGraph, index_encoding=index_encoding, bit_value=int(error_vector[iVar]))
        G.add_node(str(iVar), entity='variableNode', index_encoding=index_encoding, bit_value=int(error_vector[iVar]))
        # TODO do we need self loop:
        G.add_edge(str(iVar), str(iVar))


# TODO i ovdje index encoding?
def add_factor_nodes(G, syndrom_vector, numFactorNodes, numVariableNodes):
    for iFactor in range(numFactorNodes):
        factorNodeIndex = iFactor + numVariableNodes # index of factor node in graph G
        G.add_node(str(factorNodeIndex), entity='factorNode', syndrome=int(syndrom_vector[iFactor]))
        # TODO do we need self loop:
        G.add_edge(str(factorNodeIndex), str(factorNodeIndex))


def connect_nodes_to_second_neighbours(G, numVariableNodes, should_connect_node_to_second_neighbours):
    if should_connect_node_to_second_neighbours:
        for iVariable in range(numVariableNodes):
            connect_node_to_second_neighbours(G, str(iVariable))


def get_second_neighbors(G, node):
    return [nodeId for nodeId, pathLength in nx.single_source_shortest_path_length(G, node, cutoff=2).items() if
            pathLength == 2]


def connect_node_to_second_neighbours(G, variableNodeId):
    for neighrbVariableNodeId in get_second_neighbors(G, variableNodeId):
        G.add_edge(variableNodeId, neighrbVariableNodeId)



generate_datasets()
#transpose_error_and_syndrom_files()


