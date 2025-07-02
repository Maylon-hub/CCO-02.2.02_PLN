from utils.utils import *
from utils.graph_eval import evaluate_graph
from graph_gen.graph_gen import *
from torch_geometric.data import Data
from torch_geometric.nn import GAE
from models.gcn_model import GCN
from torch_geometric.transforms import RemoveDuplicatedEdges



import logging
logging.basicConfig(
    level=logging.INFO,  # Pode ser DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)

args = get_args()
if args.config:
    config_params = load_config_from_json(args.config)
    # Atualiza os parâmetros do argparse com os valores do JSON
    for key, value in config_params.items():
        setattr(args, key, value)

if args.gen_samples:
    data = pd.read_csv('dataset/Fact_checked_news.tsv', sep = '\t')
    generate_sample(data = data, random_state=args.gen_samples_random_state)

    logging.info("Amostra gerada no caminho dataset/sample_trabalho1.tsv")

if args.evaluate_sample:
    df = pd.read_csv('dataset/sample_trabalho1_tratada_manual.tsv', sep = '\t')
    classes = df['label']
    num_nodes = df.shape[0]


    # embedding doc2vec euclidean
    edge_index = embedding_sim_graphs(df, 'news', 'doc2vec', similarity='euclidean')

    draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_embedding_sim_graph_doc2vec_cosine.png', classes = df['label'], num_nodes=df.shape[0])

    print('MÉTRICAS DO GRAFO DOC2VEC POR SIMILARIDADE EUCLIDIANA')
    print('density ', evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'density'))
    print('assortativity', evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'assortativity'))
    print('number of edges', len(edge_index[0]) / 2)
    evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'node_degree', plot_path='results/graph_distribution/ndd_sim_doc2vec_euclid.png')


    # REM
    edge_index  = REM_graph(df, 'news', log_file='results/graph_logs/graph_log_rem.txt')

    draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_REM.png', classes = df['label'], num_nodes=df.shape[0])

    print('MÉTRICAS DO GRAFO REN')
    print('density ', evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'density'))
    print('assortativity', evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'assortativity'))
    print('number of edges', len(edge_index[0]) / 2)
    evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'node_degree', plot_path='results/graph_distribution/ndd_rem.png')

    # Yake (não deu muito certo com essa sample)
    edge_index = yake_graph(df, 'news', log_file='results/graph_logs/graph_log_yake.txt')
    draw_graph_from_edge_index(edge_index=edge_index, output_path = 'results/graph_images/graph_yake.png', classes = df['label'], num_nodes=df.shape[0])

    print('MÉTRICAS DO GRAFO YAKE')
    print('density ', evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'density'))
    print('assortativity', evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'assortativity'))
    print('number of edges', len(edge_index[0]) / 2)
    evaluate_graph(edge_index = edge_index, num_nodes=num_nodes, classes=classes, metric = 'node_degree', plot_path='results/graph_distribution/ndd_yake.png')

    #TODO: Colocar o args.benchmark para executar a tarefa completa em args.dataset_path

if args.benchmark:

    # Fazer o input por json

    df = pd.read_csv(args.benchmark_dataset_path, sep = '\t')

    x = feature_gen(df, 'news')

    if args.graph_generator == 'yake':
        edge_index = yake_graph(df, 'news')
    if args.graph_generator == 'ren':
        edge_index = REM_graph(df, 'news')
    if args.graph_generator == 'embedding':
        if args.embedding_graph_mode == 'word2vec':
            edge_index = embedding_sim_graphs(df = df, text_column= 'news', embedding='word2vec')
        if args.embedding_graph_mode == 'doc2vec':
            edge_index = embedding_sim_graphs(df = df, text_column= 'news', embedding='doc2vec')
    if args.graph_generator == 'none':
        edge_index = None

    graph_data = Data(x = x, edge_index = edge_index, y = torch.tensor(df['label']))

    # Removendo edges duplicados
    if graph_data.edge_index != None:
        transform = RemoveDuplicatedEdges()
        graph_data = transform(graph_data)

    print(graph_data)
    # print(has_duplicate_edges(graph_data.edge_index))

    # Momento de treinamento do modelo.

    # Aplicando um GAE se possível

    if graph_data.edge_index != None:
        encoder = GCN(in_channels = graph_data.x.shape[1], hidden_channels = 32, out_channels = 16)
        model = GAE(encoder = encoder)
        # TODO: colocar a learning rate (lr) e as epocas como parâmetros no args
        optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001)
        epochs = 100 
        train_gae(data = graph_data, gae_model = model, optimizer = optimizer, epochs = epochs)

        graph_data.x = model.encode(x, edge_index)

    train_and_evaluate_svm(graph_data.x, graph_data.y)

    

