import pandas as pd
import argparse

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

