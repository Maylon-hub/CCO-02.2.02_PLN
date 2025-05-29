from utils.utils import *

import logging
logging.basicConfig(
    level=logging.INFO,  # Pode ser DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s'
)

args = get_args()

if args.gen_samples:
    data = pd.read_csv('dataset/Fact_checked_news.tsv', sep = '\t')
    generate_sample(data = data, random_state=args.gen_samples_random_state)

    logging.info("Amostra gerada no caminho dataset/sample_trabalho1.tsv")