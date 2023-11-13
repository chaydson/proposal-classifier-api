import pandas as pd
import re, io
import requests
from zipfile import ZipFile
import urllib3
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skl2onnx import to_onnx
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import nltk
import unidecode 
from nltk import tokenize
from string import punctuation
import onnxruntime as rt
import joblib

nltk.download('stopwords')
nlp = spacy.load('pt_core_news_lg')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

pd.set_option('max_colwidth', None)  # Do not truncate the contents of cells in the DataFrame
REGX_HTML = r"<[^<]+?>" # Regex for HTML tags
REGX_ENDING = r'Órgão Responsável:.+' # Regex for the part of the text
POS_TAGS = ['AUX', 'ADP', 'ADV', 'CCONJ', 'DET', 'INTJ', 'NUM']

def limpeza(text):
  # text = text.lower()

  text = re.sub(REGX_HTML, '', text)  # Removendo tags HTML
  text = re.sub(REGX_ENDING, '', text)

  return text

def init():
    propostas = pd.read_csv("./app/model/production/data_extraction/brasilparticipativo.presidencia.gov.br-open-data-proposals.csv", delimiter=";")
    propostas = pd.DataFrame(propostas, columns=['category/name/pt-BR','title/pt-BR','body/pt-BR'])
    propostas.rename(columns={'category/name/pt-BR': 'Categoria', 'title/pt-BR': 'Título', 'body/pt-BR': 'Corpo'}, inplace=True)

    # Retira as linhas que estao vazias
    propostas = propostas.dropna()
    propostas.drop(propostas[propostas['Título'] == 'Tema '].index, inplace=True)  # Removendo coluna "Tema "

    # Une as colunas de Título e Corpo em uma única coluna e exclui as antigas 
    propostas['Texto'] = propostas['Título'] + '. ' + propostas['Corpo']
    propostas = propostas[['Categoria', 'Texto']]
    propostas['Texto'] = propostas['Texto'].apply(limpeza)

    simuladas = pd.read_csv("./app/model/production/data_extraction/propostas_simuladas.csv")
    propostas_simuladas = pd.DataFrame(simuladas)
    propostas = pd.concat([propostas,propostas_simuladas], ignore_index=True)
    cats = propostas['Categoria'].unique() # Pegando cada categoria única
    cats = dict(enumerate(cats, 0)) # Convertendo para dict (com índices enumerados)
    cats = {v:k for k,v in cats.items()}  # Trocando chaves e valores

    propostas['id_cats'] = propostas['Categoria'].map(cats)
    todas_palavras = ' '.join([texto for texto in propostas.Texto])

    # realiza a tokenização do texto
    token_espaco = tokenize.WhitespaceTokenizer()
    token_frase = token_espaco.tokenize(todas_palavras)

    # Adciona as stopwords em portugues
    palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese") 

    #retira as stopwords de todas as propostas
    frase_processada = list()
    for opiniao in propostas.Texto:
        nova_frase = list()
        palavras_texto = token_espaco.tokenize(opiniao)
        for palavra in palavras_texto:
            if palavra not in palavras_irrelevantes:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))

    # As propostas estao limpas das stopwords
    propostas["Texto"] = frase_processada
    # Adciona as pontuacoes 
    token_pontuacao = tokenize.WordPunctTokenizer()

    pontuacao = list()
    for ponto in punctuation:
        pontuacao.append(ponto)

    pontuacao_stopwords = pontuacao + palavras_irrelevantes

    frase_processada = list()
    for opiniao in propostas["Texto"]:
        nova_frase = list()
        palavras_texto = token_pontuacao.tokenize(opiniao)
        for palavra in palavras_texto:
            if palavra not in pontuacao_stopwords:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))

    propostas["Texto"] = frase_processada
    sem_acentos = [unidecode.unidecode(texto) for texto in propostas["Texto"]]

    stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]

    propostas["Texto"] = sem_acentos

    frase_processada = list()
    for opiniao in propostas["Texto"]:
        nova_frase = list()
        palavras_texto = token_pontuacao.tokenize(opiniao)
        for palavra in palavras_texto:
            if palavra not in pontuacao_stopwords:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))

    propostas["Texto"] = frase_processada

    #deixa todas as letras minusculas
    frase_processada = list()
    for opiniao in propostas["Texto"]:
        nova_frase = list()
        opiniao = opiniao.lower()
        palavras_texto = token_pontuacao.tokenize(opiniao)
        for palavra in palavras_texto:
            if palavra not in stopwords_sem_acento:
                nova_frase.append(palavra)
        frase_processada.append(' '.join(nova_frase))

    propostas["Texto"] = frase_processada
    docs = list(propostas['Texto'])

    # A próxima linha configura o vetorizador TF-IDF com uso de IDF (Inverse Document Frequency) e um limite máximo de recursos em 20,000.
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2),use_idf=True, max_features=50000)

    # Aqui, estamos transformando os documentos da lista 'docs' em vetores TF-IDF.
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)

    # Finalmente, estamos convertendo os vetores TF-IDF em uma matriz NumPy e armazenando-a na variável 'docs'.
    docs = tfidf_vectorizer_vectors.toarray()
    X = docs

    # Atribui a variável y à coluna 'id_cats' do DataFrame 'propostas'
    y = propostas['Categoria']

    # Imprime as dimensões (formato) de X e y, ou seja, o número de linhas e colunas
    SEED = 123

    # Dividindo o conjunto de dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)    
    svc = LinearSVC(class_weight='balanced')
    svc.fit(X_train, y_train)
    onx = to_onnx(svc, X_train[:1].astype(np.float32))
    with open("./app/model/proposal_classifier_v1_1_0.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    joblib.dump(tfidf_vectorizer, './app/model/tfidf_vectorizer.joblib')

def run_model(text):
    loaded_tfidf_vectorizer = joblib.load('./app/model/tfidf_vectorizer.joblib')
    text = [text]
    text = loaded_tfidf_vectorizer.transform(text).toarray().astype(np.float32)

    sess = rt.InferenceSession("./app/model/proposal_classifier_v1_1_0.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred_onx = sess.run([label_name], {input_name: text})[0]

    return pred_onx[0]
