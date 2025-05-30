from graph_gen.graph_gen import *
import pandas as pd
from utils.utils import *

df = pd.read_csv('dataset/sample_trabalho1_tratada_manual.tsv', sep = '\t')

# embedding doc2vec euclidean
edge_index = embedding_sim_graphs(df, 'news', 'doc2vec', similarity='euclidean')

draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_embedding_sim_graph_doc2vec_cosine.png')

# REM
edge_index  = REM_graph(df, 'news')

draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_REM.png')

# Yake (não deu muito certo com essa sample)
edge_index = yake_graph(df, 'news')
draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_yake.png')