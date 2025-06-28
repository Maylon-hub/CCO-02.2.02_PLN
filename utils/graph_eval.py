import networkx as nx
import matplotlib.pyplot as plt
import os

def evaluate_graph(edge_index, num_nodes, classes, metric='density', plot_path='degree_distribution.png'):
    """
    Avalia métricas de um grafo a partir do edge_index.

    Parâmetros:
    -----------
    edge_index : torch.Tensor
        Tensor [2, num_edges] com as conexões do grafo.

    metric : str
        Métrica a ser calculada: 'assortativity', 'node_degree', 'modularity' ou 'density'.

    plot_path : str
        Caminho do arquivo para salvar o gráfico de distribuição de grau (apenas para 'node_degree').

    Retorna:
    --------
    Se metric == 'node_degree': caminho do gráfico salvo (str).
    Caso contrário: valor da métrica (float).
    """
    # Constrói o grafo
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)

    for idx, c in enumerate(classes):
        G.nodes[idx]['classe'] = c

    if metric == 'assortativity':
        # Assortatividade por grau
        value = nx.degree_assortativity_coefficient(G, 'classe')
        return value

    elif metric == 'density':
        # Densidade do grafo
        value = nx.density(G)
        return value

    elif metric == 'modularity':
        try:
            import community as community_louvain
        except ImportError:
            raise ImportError("A biblioteca 'python-louvain' é necessária para calcular a modularidade. Instale-a via pip.")
        # Calcula a partição das comunidades usando Louvain
        partition = community_louvain.best_partition(G)
        # Calcula a modularidade
        modularity = community_louvain.modularity(partition, G)
        return modularity

    elif metric == 'node_degree':
        # Distribuição de grau
        degrees = [degree for node, degree in G.degree()]
        plt.figure(figsize=(8, 6))
        plt.hist(degrees, bins=range(1, max(degrees)+2), align='left', edgecolor='black')
        plt.xlabel("Grau do nó")
        plt.ylabel("Número de nós")
        # plt.title("Distribuição de Grau dos Nós")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Salva o gráfico
        plt.savefig(plot_path)
        plt.close()
        return os.path.abspath(plot_path)

    else:
        raise ValueError("Métrica não reconhecida. Use: 'assortativity', 'node_degree', 'modularity' ou 'density'.")
