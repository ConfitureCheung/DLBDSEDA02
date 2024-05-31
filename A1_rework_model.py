import pandas as pd
import numpy as np
import time
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim.corpora as corpora
from gensim import models
from gensim.models import LsiModel
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel


def start_program():

    return time.time()


def end_program(start_time):
    end_time = time.time()
    seconds = end_time - start_time
    minutes = seconds // 60
    hours = minutes // 60
    print("%02d:%02d:%02d" % (hours, minutes % 60, seconds % 60))


# Part 1) Data Loading from Kaggle.com
def load_kaggle_df():
    path_in = 'data'
    df = pd.read_csv(f'{path_in}\\Consumer_Complaints.csv')
    df = df[['Consumer complaint narrative', 'Product']]
    # print(df.columns)
    # print(df.dtypes)
    # print(df.shape)  # (903983, 2)
    print(df['Product'].value_counts())
    print('=' * 80)

    return df


# Part 2) Data Cleansing
def df_data_cleansing(df):
    # Remove NA values
    df1 = df.dropna(subset=['Consumer complaint narrative'])
    print(df1.shape)  # (199970, 2)

    # Check Number of Topics (Original)
    df1['Product id'] = df1['Product'].factorize()[0]
    # df1.to_csv('df1.csv')

    pdt_id_df = df1[['Product', 'Product id']].drop_duplicates().sort_values('Product id')
    print(pdt_id_df)
    print(pdt_id_df.shape)  # (18, 2)
    print('=' * 80)

    return df1


# Part 6) Model Evaluation with new model method
def clean_text(text):
    tokens = word_tokenize(text)

    stop_words = set(nltk.corpus.stopwords.words('english'))

    tokens = [word for word in tokens if word.lower() not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens


def model_evaluation(df):
    df['clean text'] = df['Consumer complaint narrative'].apply(lambda x: clean_text(text=x))

    dictionary = corpora.Dictionary(df['clean text'])
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in df['clean text']]
    tfidf = models.TfidfModel(doc_term_matrix)
    doc_term_matrix_tfidf = tfidf[doc_term_matrix]

    ''' further breakdown below codes to new func if works '''
    lda_model = models.ldamodel.LdaModel
    ldamodel = lda_model(doc_term_matrix_tfidf, num_topics=18, id2word=dictionary, passes=80, random_state=42)

    # lsa_model = LsiModel()
    # lsamodel = lsa_model(common_corpus[:3], id2word=common_dictionary)

    # nmf_model = models.Nmf()
    # nmfmodel = nmf_model(doc_term_matrix, num_topics=18)

    coherence_model_lda = CoherenceModel(model=ldamodel, texts=df['clean text'], dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


def main():
    start_time = start_program()

    # Part 1
    df = load_kaggle_df()
    # Part 2
    df1 = df_data_cleansing(df=df)

    # Part 6
    model_evaluation(df=df1)

    end_program(start_time)


# Manual Run
# ----------------------------------------------------
if __name__ == "__main__":
    main()