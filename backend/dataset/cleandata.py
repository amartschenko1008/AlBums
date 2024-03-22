'''import json

filename = 'booksummaries.txt'

books = {}

parts = ['articleID', 'freebaseID', 'title' 'author', 'pubDate', 'genres', 'summary']

with open(filename) as fh:
    for line in fh:
        desc = line.strip().split("\\s{2,}")
        dict2 = {}
        i = 0
        print(desc[1])
        while i < len(parts):
            if parts[i] != 'title':
                dict2[parts[i]] = desc[i]
            i += 1
        books[desc[3]] = dict2

out_file = open("test.json", "w")
json.dump(books, out_file, indent = 4)
out_file.close()'''



import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

column_names = ["wikipedia_id", "freebase_id", "book_title","book_author","book_pub_date",
                "book_genres","plot_summary"]
cmu_db = pd.read_table("booksummaries.txt", names=column_names)
cmu_db = cmu_db.drop_duplicates(subset="book_title", ignore_index= True)


#From A5
def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    """Returns a TfidfVectorizer object with the above preprocessing properties.

    Note: This function may log a deprecation warning. This is normal, and you
    can simply ignore it.

    Parameters
    ----------
    max_features : int
        Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer
        constructer.
    stop_words : str
        Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer.
    max_df : float
        Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer.
    min_df : float
        Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer.
    norm : str
        Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer.

    Returns
    -------
    TfidfVectorizer
        A TfidfVectorizer object with the given parameters as its preprocessing properties.
    """
    # TODO-5.1

    vectorizer = TfidfVectorizer(max_df=max_df,min_df=min_df,
    max_features=max_features,stop_words=stop_words, norm=norm)

    return vectorizer

book_title_to_idx = {}
book_idx_to_title = {}

for index, row in cmu_db.iterrows():
  book_title_to_idx[row['book_title']] = index
  book_idx_to_title[index] = row['book_title']

assert len(cmu_db) == len(book_title_to_idx) == len(book_idx_to_title)
n_feats = 5000
doc_by_vocab = np.empty([len(cmu_db), n_feats])
tfidf_vec = build_vectorizer(n_feats, "english")
doc_by_vocab = tfidf_vec.fit_transform([row['plot_summary'] for index, row in cmu_db.iterrows()]).toarray()
index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names_out())}

netflix_db = pd.read_csv("titles.csv")
netflix_db = netflix_db.drop_duplicates(subset="title", ignore_index=True)
netflix_db = netflix_db.fillna("")
# @title type

netflix_title_to_idx = {}
netflix_idx_to_title = {}

for index, row in netflix_db.iterrows():
  netflix_title_to_idx[row['title']] = index
  netflix_idx_to_title[index] = row['title']
assert len(netflix_title_to_idx) == len(netflix_idx_to_title) == len(netflix_db)
query_by_vocab = np.empty([len(netflix_db), n_feats])

query_by_vocab = tfidf_vec.transform([row['description'] for index, row in netflix_db.iterrows()]).toarray()
from sklearn.metrics.pairwise import cosine_similarity
def get_sim_book(netflix_title, book_mat):
  if netflix_title in netflix_title_to_idx:
    netflix_idx = netflix_title_to_idx[netflix_title]
    netflix_vec = query_by_vocab[netflix_idx].reshape(1,-1)

    similarities = cosine_similarity(netflix_vec, book_mat)

    return similarities
similarities = get_sim_book("Breaking Bad", doc_by_vocab)

def book_sims_to_recs(book_sims, book_idx_to_title):
  sim_pairs = [(book_idx_to_title[i], sim) for i, sim in enumerate(book_sims[0])]

  top_5 = sorted(sim_pairs, key=lambda x: x[1], reverse = True)[:5]

  return top_5


def rec_books(netflix_title, book_mat, book_idx_to_title):
  similarities = get_sim_book(netflix_title, book_mat)

  top_5 = book_sims_to_recs(similarities, book_idx_to_title)

  print(f"The 5 most similar books to {netflix_title} are:")
  for index, (book_title, sim_score) in enumerate(top_5):
    print(f"\n{index+1}. {book_title}")