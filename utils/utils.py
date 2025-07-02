import pandas as pd
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def get_args():
    parser = argparse.ArgumentParser(description="Descrição do seu programa.")

    parser.add_argument('--config', type=str, help='Path to the JSON config file with parameters')

    parser.add_argument('--gen_samples', action='store_true', help='Chama a função de geração de amostras')
    parser.add_argument('--gen_samples_random_state', type=int, default=42, help='random_state da geração de amostra')
    parser.add_argument('--evaluate_sample', action = 'store_true', help = 'Avalia a sample gerada por --gen_sample')

    # Main Benchmark
    parser.add_argument('--benchmark', action='store_true', help = 'Chama a função principal para executar o benchmark')
    parser.add_argument('--graph_generator', type = str, help = 'Gerador de grafos utilizado no benchmark')
    parser.add_argument('--embedding_graph_mode', type = str, default='word2vec', help = 'modo de embedding da estratégia de gerar o grafo por embedding, podendo ser word2vec ou doc2vec')

    parser.add_argument('--benchmark_dataset_path', type=str, help = 'Caminho do dataset (csv) para fazer o benchmark principal')
    parser.add_argument('--embedding_path', type=str, help = 'caminho dos embeddings, se não tiver, faz embeddings com Doc2Vec')

    

    args = parser.parse_args()
    return args


def load_config_from_json(json_file):
    '''Function to load parameters from a JSON file'''
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

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
    G.to_undirected()
    G.add_edges_from(edges)
    
    # Mapeamento de cores: 0 → azul, 1 → vermelho
    color_map = ['blue' if c == -1 else 'red' for c in classes]

    # Desenho do grafo
    plt.figure(figsize=(8, 6))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G, pos, with_labels=True, 
        node_color=color_map, 
        edge_color='gray', 
        node_size=700, 
        font_size=20,
        # alpha = 0.7
    )
    plt.title("Grafo gerado a partir do edge_index")
    plt.axis('off')

    # Salva a imagem
    plt.savefig(output_path, format='png')
    plt.close()

def has_duplicate_edges(edge_index: torch.Tensor) -> bool:
    """
    Verifica se há arestas duplicadas no edge_index (direcionado).

    Parâmetros:
    - edge_index: Tensor shape (2, num_edges)

    Retorna:
    - True se houver arestas duplicadas, False caso contrário
    """
    # Transforma em colunas (pares de arestas)
    edges = edge_index.t()  # shape: (num_edges, 2)
    
    # Converte para uma lista de tuplas para usar em set
    edge_tuples = [tuple(edge.tolist()) for edge in edges]
    
    # Compara o número total de arestas com o número de arestas únicas
    return len(edge_tuples) != len(set(edge_tuples))

def train_gae(data, gae_model, optimizer, epochs, verbose = True):
    for e in range(epochs):
            optimizer.zero_grad()
            H_L = gae_model.encode(data.x.float(), data.edge_index)
            loss = gae_model.recon_loss(H_L, data.edge_index)
            if verbose:
                print(f'epoch {e+1} | loss {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()
    return

def train_and_evaluate_svm(x: torch.Tensor, y: torch.Tensor, test_size=0.5, random_seed=100):
    """
    Divide os dados em treino e teste, treina uma SVM e avalia usando F1-score e outras métricas.

    Parâmetros:
    - x (torch.Tensor): Features numéricas (shape: [n amostras, n features])
    - y (torch.Tensor): Rótulos/classes (shape: [n amostras])
    - test_size (float): Proporção do conjunto de teste
    - random_seed (int): Semente para reprodutibilidade

    Retorna:
    - None (imprime as métricas)
    """

    # Converte para numpy
    X_np = x.detach().numpy()
    y_np = y.detach().numpy()

    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_seed, stratify=y_np
    )

    # Cria e treina a SVM
    clf = SVC(kernel='rbf', random_state=random_seed)
    clf.fit(X_train, y_train)

    # Faz previsões
    y_pred = clf.predict(X_test)

    # Avaliação
    print("=== Matriz de Confusão ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Relatório de Classificação ===")
    print(classification_report(y_test, y_pred))

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\nF1 Score (ponderado): {f1:.4f}")