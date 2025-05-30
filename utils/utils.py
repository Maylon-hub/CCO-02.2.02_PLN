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

def draw_graph_from_edge_index(edge_index, output_path):
    """
    Desenha um grafo a partir de um edge_index e salva como imagem no caminho especificado.

    Parâmetros:
    -----------
    edge_index : torch.Tensor
        Tensor no formato [2, num_edges] contendo as arestas do grafo.

    output_path : str
        Caminho do arquivo de saída para salvar a imagem do grafo (ex: "grafo.png").
    """
    # Verificação básica
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index deve ter o formato [2, num_edges].")

    # Criação do grafo
    G = nx.Graph()
    edges = edge_index.t().tolist()  # lista de tuplas (i, j)
    G.add_edges_from(edges)

    # Desenho do grafo
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # layout automático para visualização
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Grafo gerado a partir do edge_index")
    plt.axis('off')
    
    # Salva a imagem
    plt.savefig(output_path, format='png')
    plt.close()