# coding=utf-8
"""
采用ARIMA和结构相似混合方法
"""

import argparse
import networkx as nx
from typing import List
from networkx import Graph
import arima
import pandas as pd


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='email-Eu-core-temporal.txt',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='email-Eu-core-temporal.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is weighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(slice_count):
    """
    Reads the input network in networkx.
    """
    col_index = ["x", "y", "time"]
    result = pd.read_table('email-Eu-core-temporal-Dept3.txt', sep=' ', header=None, names=col_index)
    list_G = []  # type: List[Graph]
    set_edge = set()
    max_time = result['time'].max()
    slice_num = max_time / slice_count + 1
    G_test = nx.Graph()
    for i in range(1, slice_count + 1):
        G = nx.Graph()
        # 添加所有节点到图中
        G.add_nodes_from(result['x'].tolist())
        G.add_nodes_from(result['y'].tolist())
        # 获取某个时间切片所有节点对
        edge = result[(result['time'] >= (i - 1) * slice_num) & (result['time'] < i * slice_num)].iloc[:, 0:2]
        # 统计出现频率作为边权重
        weighted_edge = edge.groupby(['x', 'y']).size().reset_index()
        weighted_edge.rename(columns={0: 'frequency'}, inplace=True)
        weighted_edge_tuples = [tuple(xi) for xi in weighted_edge.values]
        G.add_weighted_edges_from(weighted_edge_tuples)
        # 测试集
        if i == slice_count:
            G_test = G
            for edge in G_test.edges():
                if G_test[edge[0]][edge[1]]['weight'] > 1:
                    G_test[edge[0]][edge[1]]['weight'] = 1
            continue
        edge_tuples = set(tuple(xi) for xi in edge.values)
        set_edge = edge_tuples | set_edge
        list_G.append(G)
        # if args.weighted:
        #     G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        # else:
        #     G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        #     for edge in G.edges():
        #         G[edge[0]][edge[1]]['weight'] = 1
        #
        # if not args.directed:
        #     G = G.to_undirected()
    G = nx.Graph()
    # 添加所有节点到图中
    G.add_nodes_from(result['x'].tolist())
    G.add_nodes_from(result['y'].tolist())
    G.add_edges_from(set_edge)
    return list_G, G, G_test


def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    list_G, nx_G, G_test = read_graph(20)
    G = arima.Graph(nx_G, list_G, G_test, args.directed)
    G.predict(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
