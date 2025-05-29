# CCO-02.2.02_PLN
Repositório das atividades I e II da disciplina CCO 02.2.02 - Processamento de Linguagem Natural

O trabalho será dividido em duas tarefas, a primeira _pré-processamento dos dados em texto_ e a segunda _geração de grafos de conhecimento_

## Pré-Processamento

Tarefa 1: Limpeza dos dados - Remoção de links, remoção de palavras que relacionam a veracidade com a semantica (Ex: verdadeiro, falso, #fake, etc.) 

## Grafos de Conhecimento

Tarefa 2: Gerar um grafo onde vértices são os textos e arestas são algumas relações entre eles, por enquanto as ideias são:

-> Unir os textos pelos embeddings (d2v, w2v, LLM) + similaridade (cos, euclidiana)
-> Unir textos pelas Keywords (YAKE, TF-IDF, RAKE)
-> Unir os textos a partir de reconhecimento de entidades nomeadas
-> Unir textos por sinônimos
-> Unir textos por anotação de papéis semânticos

# Trabalho 1

Para o seminário 1 a ideia é aplicar uma sample dos dados. Selecionar uns 10 textos, limpá-los manualmente e comparar com algumas das técnicas, apresentando os resultados. A mesma coisa para a geração de grafos, gerar grafos com essas 10 samples e avaliar quantidade de arestas, componentes conexas, etc.