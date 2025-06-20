# utils/graph_tools.py

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def visualize_interaction_graph(relation_matrix, node_labels=None, save_path=None):
    """
    📌 의미 상호작용 관계 행렬을 기반으로 그래프 시각화
    - relation_matrix: (N x N) numpy array
    - node_labels: list of node names (length N)
    - save_path: 저장할 경로 (없으면 show)
    """
    N = relation_matrix.shape[0]
    G = nx.Graph()

    # 노드 추가
    for i in range(N):
        label = node_labels[i] if node_labels else f"Obj_{i}"
        G.add_node(i, label=label)

    # 엣지 추가
    for i in range(N):
        for j in range(i+1, N):
            weight = relation_matrix[i, j]
            if weight > 0.01:  # 임계값 이상만
                G.add_edge(i, j, weight=weight)

    pos = nx.spring_layout(G)
    edge_weights = [G[u][v]['weight'] for u,v in G.edges()]

    # 시각화
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=edge_weights)
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['label'] for i in G.nodes})
    plt.title("Semantic Interaction Graph")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ 그래프 저장됨: {save_path}")
    else:
        plt.show()