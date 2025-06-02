from graph_gen.graph_gen import *
import pandas as pd
from utils.utils import *
from utils.graph_eval import evaluate_graph

df = pd.read_csv('dataset/sample_trabalho1_tratada_manual.tsv', sep = '\t')

# embedding doc2vec euclidean
edge_index = embedding_sim_graphs(df, 'news', 'doc2vec', similarity='euclidean')

draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_embedding_sim_graph_doc2vec_cosine.png', classes = df['label'], num_nodes=df.shape[0])

# REM
edge_index  = REM_graph(df, 'news', log_file='results/graph_logs/graph_log_rem.txt')

draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_REM.png', classes = df['label'], num_nodes=df.shape[0])

# Yake (não deu muito certo com essa sample)
edge_index = yake_graph(df, 'news', log_file='results/graph_logs/graph_log_yake.txt')
draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_yake.png', classes = df['label'], num_nodes=df.shape[0])

print(evaluate_graph(edge_index))