import pandas as pd
import numpy as np
import time
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF


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


# Part 3) Text Preprocessing (BOW / TF-IDF)
def bow_approach(df):
    print('BOW approach')
    reviews_list = df['Consumer complaint narrative'][0:2].values.tolist()
    vect = CountVectorizer()
    data = vect.fit_transform(reviews_list)
    bow_data = pd.DataFrame(data.toarray(), columns=vect.get_feature_names_out())
    bow_data.to_csv('bow_data.csv', index=False)
    print(bow_data)
    print('='*80)


def tfidf_approach(df):
    print('TF-IDF approach')
    # min_df is used for removing terms that appear too infrequently.
    # e.g. min_df = 0.01 --> "ignore terms that appear in less than 1% of the documents"
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2',
                            encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    # Transform each complaint into a vector
    reviews_list = df['Consumer complaint narrative'][0:2]
    features = tfidf.fit_transform(reviews_list).toarray()
    tfidf_data = pd.DataFrame(features, columns=tfidf.get_feature_names_out())
    tfidf_data.to_csv('tfidf_data.csv', index=False)
    print(tfidf_data)
    print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))
    print('='*80)


# Part 4) Semantic Analysis (LSA / LDA / NMF)
def model_creation(df):
    # using tf-idf approach to build the model
    # max_features=18 according to the 'Product' count
    # vectorizer = TfidfVectorizer(use_idf=True, max_features=18, smooth_idf=True, stop_words='english')
    vectorizer = TfidfVectorizer(use_idf=True, max_features=18, smooth_idf=True,
                                 norm='l2', encoding='latin-1', stop_words='english')
    model = vectorizer.fit_transform(df['Consumer complaint narrative'])

    return model, vectorizer


# 1) Latent semantic analysis (LSA) approach
def lsa_approach(model):
    # n_components=18 according to the 'Product' count
    LSA_model = TruncatedSVD(n_components=18, algorithm='randomized', n_iter=10)
    lsa = LSA_model.fit_transform(model)

    return lsa, LSA_model


# 2) Latent Dirichlet Allocation (LDA) approach
def lda_approach(model):
    # n_components=18 according to the 'Product' count
    LDA_model = LatentDirichletAllocation(n_components=18, learning_method='online', random_state=42, max_iter=10)
    lda = LDA_model.fit_transform(model)

    return lda, LDA_model


# 3) Non-Negative Matrix Factorization (NMF) approach
def nmf_approach(model):
    # n_components=18 according to the 'Product' count
    NMF_model = NMF(n_components=18, init='random', random_state=42, max_iter=10)
    nmf = NMF_model.fit_transform(model)

    return nmf, NMF_model


# Comment-Topic Matrix
def demonstrate_result1(df, approach):
    idx_list, val_list = [], []

    print(f'Demo, One Review via {approach} :')
    for i, topic in enumerate(df[0]):
        print("Topic ", i, " : ", topic * 100)
    print('=' * 80)

    # Full Review of Semantic Analysis
    for comment in df:
        topic_val = np.amax(comment, axis=0)
        topic_idx = comment.argmax(axis=0)
        # print(f'Topic {topic_idx} : {topic_val}')
        topic_idx = f'Topic {topic_idx}'
        idx_list.append(topic_idx)
        val_list.append(topic_val)

    return idx_list, val_list


# Document-Term Matrix
def demonstrate_result2(vect, model, approach):
    print(f'Demo, One Review via {approach} :')

    terms = vect.get_feature_names_out()

    for i, comp in enumerate(model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:10]
        result = ', '.join([t[0] for t in sorted_terms])
        print("Topic " + str(i) + ": " + result)
    print('=' * 80)


def result_to_csv(df):
    model, vectorizer = model_creation(df=df)
    lsa, LSA_model = lsa_approach(model=model)
    lda, LDA_model = lda_approach(model=model)
    nmf, NMF_model = nmf_approach(model=model)

    lsa_topic_idx_list, lsa_topic_val_list = demonstrate_result1(df=lsa, approach='LSA')
    lda_topic_idx_list, lda_topic_val_list = demonstrate_result1(df=lda, approach='LDA')
    nmf_topic_idx_list, nmf_topic_val_list = demonstrate_result1(df=nmf, approach='NMF')

    demonstrate_result2(vect=vectorizer, model=LSA_model, approach='LSA')
    demonstrate_result2(vect=vectorizer, model=LDA_model, approach='LDA')
    demonstrate_result2(vect=vectorizer, model=NMF_model, approach='NMF')

    # convert results to df
    data = {'LSA Topic': lsa_topic_idx_list, 'LSA Val': lsa_topic_val_list,
            'LDA Topic': lda_topic_idx_list, 'LDA Val': lda_topic_val_list,
            'NMF Topic': nmf_topic_idx_list, 'NMF Val': nmf_topic_val_list}

    lsa_lda_nmf_df = pd.DataFrame(data)
    print(lsa_lda_nmf_df.shape)  # (199970, 4)

    df_merge = pd.concat([df, lsa_lda_nmf_df], axis=1)
    df_merge.to_csv('lsa_lda_nmf.csv')

    return df_merge, model, lsa, LSA_model, lda, LDA_model, nmf, NMF_model


# Part 5) Visualization via Plotly
def plotly_hbarchart(df, gpby_col, y_col, title_name):
    df_group = df.groupby(y_col)[gpby_col].count().reset_index(name='Count')
    df_group = df_group.sort_values(by=['Count'])
    # df_group = df_group.nlargest(30, 'Count')
    fig = px.bar(df_group, x='Count', y=y_col, text_auto='.2s', orientation='h', title=f'{title_name} Topic Classification')
    fig.show()


def main():
    start_time = start_program()

    # Part 1
    df = load_kaggle_df()
    # Part 2
    df1 = df_data_cleansing(df=df)
    # Part 3
    bow = bow_approach(df=df1)
    tfidf = tfidf_approach(df=df1)
    # Part 4
    df_merge, model, lsa, LSA_model, lda, LDA_model, nmf, NMF_model = result_to_csv(df=df1)
    # Part 5
    plotly_hbarchart(df=df_merge, gpby_col='Consumer complaint narrative', y_col='Product', title_name='Original')
    plotly_hbarchart(df=df_merge, gpby_col='LSA Topic', y_col='LSA Topic', title_name='LSA')
    plotly_hbarchart(df=df_merge, gpby_col='LDA Topic', y_col='LDA Topic', title_name='LDA')
    plotly_hbarchart(df=df_merge, gpby_col='NMF Topic', y_col='NMF Topic', title_name='NMF')

    end_program(start_time)


# Manual Run
# ----------------------------------------------------
if __name__ == "__main__":
    main()