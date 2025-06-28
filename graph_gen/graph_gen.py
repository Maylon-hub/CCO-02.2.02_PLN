'''
Arquivo com os algoritmos de geração de grafos
'''

from itertools import combinations
import yake
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import spacy

def REM_graph(df, text_column='news', threshold=1, log_file='graph_connections.txt'):
    """
    Gera um edge_index conectando textos que compartilham entidades nomeadas e salva um arquivo
    de justificativa para cada aresta.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os textos.

    text_column : str
        Nome da coluna com os textos (padrão: 'news').

    threshold : int
        Número mínimo de entidades nomeadas em comum para criar uma aresta.

    log_file : str
        Nome do arquivo de saída com as justificativas.

    Retorna:
    --------
    edge_index : torch.Tensor
        Tensor [2, num_edges] com as conexões entre os textos baseado em entidades nomeadas.
    """
    nlp = spacy.load("pt_core_news_sm")

    # Extrai entidades nomeadas de cada texto
    entity_sets = []
    for text in df[text_column]:
        doc = nlp(str(text))
        entities = set(ent.text.strip().lower() for ent in doc.ents)
        entity_sets.append(entities)

    # Inicializa o arquivo de justificativas
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Justificativas de conexões no grafo:\n\n")

        # Constrói lista de arestas e salva justificativas
        edges = []
        for i, j in combinations(range(len(entity_sets)), 2):
            common_entities = entity_sets[i] & entity_sets[j]
            if len(common_entities) >= threshold:
                edges.append((i, j))
                edges.append((j, i))  # Grafo não direcionado

                # Salva justificativa
                f.write(f"Conexão entre {i} e {j} - Entidades em comum: {', '.join(common_entities)}\n")

    if not edges:
        raise ValueError("Nenhuma conexão encontrada com o threshold especificado.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def yake_graph(df, text_column, log_file='yake_graph_connections.txt'):
    """
    Gera um grafo entre textos baseado em sobreposição de palavras-chave extraídas pelo YAKE e
    salva um arquivo de justificativas.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os textos a serem analisados.

    text_column : str
        Nome da coluna do DataFrame que contém os textos.

    log_file : str
        Nome do arquivo de saída para salvar as justificativas.

    Retorna:
    --------
    edge_index : torch.Tensor
        Tensor [2, num_edges] com as conexões entre os textos baseadas em palavras-chave relevantes.
    """
    def extract_keywords(text, ngram_range=(1, 3), top_k=15):
        # Extrai n-gramas (até trigramas)
        custom_kw_extractor = yake.KeywordExtractor(
            lan="pt",
            n=ngram_range[1],
            top=top_k,
            features=None
        )
        keywords = custom_kw_extractor.extract_keywords(text)
        # Filtra apenas n-gramas desejados (1 a 3 palavras)
        selected = [kw for kw, score in keywords if len(kw.split()) in range(ngram_range[0], ngram_range[1]+1)]
        return set(selected)

    # Passo 1: extrair n-gramas relevantes
    keyword_sets = df[text_column].apply(lambda txt: extract_keywords(txt)).tolist()

    # Inicializa o arquivo de justificativas
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("Justificativas de conexões no grafo (baseadas em palavras-chave YAKE):\n\n")

        # Passo 2: comparar cada par e gerar justificativas
        edges = []
        for i, j in combinations(range(len(keyword_sets)), 2):
            common_keywords = keyword_sets[i] & keyword_sets[j]
            if common_keywords:  # interseção não vazia
                edges.append((i, j))
                edges.append((j, i))  # grafo não direcionado
                f.write(f"Conexão entre {i} e {j} - Palavras-chave em comum: {', '.join(common_keywords)}\n")

    if not edges:
        raise ValueError("Nenhuma conexão entre os textos encontrada com os critérios definidos.")

    # Passo 3: transformar em edge_index
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index   

def embedding_sim_graphs(df, text_column, embedding='word2vec', similarity='cosine', k_neighbors=3):
    # TODO: fazer um embedding usando bert
    """
    Gera um grafo a partir de embeddings dos textos com base em k vizinhos mais próximos.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame contendo os textos.

    text_column : str
        Nome da coluna com os textos.

    embedding : str
        Tipo de embedding: 'word2vec', 'doc2vec' ou 'llm'.

    similarity : str
        Métrica: 'cosine' ou 'euclidean'.

    k_neighbors : int
        Número de vizinhos mais próximos a conectar para cada texto.

    Retorna:
    --------
    edge_index : torch.Tensor
        Tensor [2, num_edges] com as conexões do grafo.
    """
    texts = df[text_column].astype(str).tolist()

    # Geração dos embeddings
    if embedding == 'word2vec':
        tokenized = [text.split() for text in texts]
        w2v_model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
        vectors = np.array([
            np.mean([w2v_model.wv[word] for word in words if word in w2v_model.wv], axis=0)
            if any(word in w2v_model.wv for word in words) else np.zeros(w2v_model.vector_size)
            for words in tokenized
        ])

    elif embedding == 'doc2vec':
        tagged_docs = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]
        d2v_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
        d2v_model.build_vocab(tagged_docs)
        d2v_model.train(tagged_docs, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
        vectors = np.array([d2v_model.infer_vector(doc.words) for doc in tagged_docs])

    elif embedding == 'llm':
        raise NotImplementedError("Processamento com LLM ainda não foi implementado.")

    else:
        raise ValueError("Tipo de embedding inválido. Use 'word2vec', 'doc2vec' ou 'llm'.")

    # Cálculo da matriz de similaridade/distância
    if similarity == 'cosine':
        sim_matrix = cosine_similarity(vectors)
        # Para ordenação correta de similaridade (maior = mais próximo)
        order = np.argsort(-sim_matrix, axis=1)
    elif similarity == 'euclidean':
        sim_matrix = euclidean_distances(vectors)
        # Para ordenação correta de distância (menor = mais próximo)
        order = np.argsort(sim_matrix, axis=1)
    else:
        raise ValueError("Métrica de similaridade inválida. Use 'cosine' ou 'euclidean'.")

    # Construção das arestas usando k vizinhos mais próximos (ignorando auto-conexões)
    edges = []
    n = len(vectors)
    for i in range(n):
        neighbors = order[i, 1:k_neighbors+1]  # ignora o próprio nó [0]
        for j in neighbors:
            edges.append((i, j))
            edges.append((j, i))  # grafo não direcionado

    if not edges:
        raise ValueError("Nenhuma conexão encontrada com os k vizinhos especificados.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, vectors