import pandas as pd
import argparse
import matplotlib.pyplot as plt
import networkx as nx

def get_args():
    parser = argparse.ArgumentParser(description="Descrição do seu programa.")

    parser.add_argument('--gen_samples', action='store_true', help='Chama a função de geração de amostras')
    parser.add_argument('--gen_samples_random_state', type=int, default=42, help='random_state da geração de amostra')

    args = parser.parse_args()
    return args


def generate_sample(data, random_state = 42, nsample_true = 5, nsample_false = 5):
    '''
    Gera a sample a ser analisada pelo trabalho 1. Salva o arquivo em dataset/sample.tsv
    '''
    # data = pd.read_csv('dataset/Fact_checked_news.tsv', sep = '\t')

    sample1 = data[data['label'] == 1].sample(n = nsample_true, random_state=random_state)
    sample2 = data[data['label'] == -1].sample(n = nsample_false, random_state=random_state)

    df = pd.concat([sample1, sample2], ignore_index = True)

    df.to_csv('dataset/sample_trabalho1.tsv', sep = '\t')


def draw_graph_from_edge_index(edge_index, output_path, classes, num_nodes):
    """
    Desenha um grafo a partir de um edge_index, incluindo vértices isolados, e salva como imagem.

    Parâmetros:
    -----------
    edge_index : torch.Tensor
        Tensor no formato [2, num_edges] contendo as arestas do grafo.

    output_path : str
        Caminho do arquivo de saída para salvar a imagem do grafo (ex: "grafo.png").
    
    classes : list ou array
        Lista ou array contendo a classe de cada nó (0 ou 1), usado para colorir os nós.
    
    num_nodes : int
        Número total de nós no grafo, incluindo vértices isolados.
    """
    # Verificação básica
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index deve ter o formato [2, num_edges].")
    if len(classes) != num_nodes:
        raise ValueError("O comprimento de 'classes' deve corresponder ao número de nós (num_nodes).")
    
    # Criação do grafo
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))  # Inclui todos os vértices, até mesmo isolados
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    
    # Mapeamento de cores: 0 → azul, 1 → vermelho
    color_map = ['blue' if c == -1 else 'red' for c in classes]

    # Desenho do grafo
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos, with_labels=True, 
        node_color=color_map, 
        edge_color='gray', 
        node_size=500, 
        font_size=10,
        alpha = 0.7
    )
    plt.title("Grafo gerado a partir do edge_index")
    plt.axis('off')

    # Salva a imagem
    plt.savefig(output_path, format='png')
    plt.close()
