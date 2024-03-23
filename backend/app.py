import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    episodes_df = pd.DataFrame(data['episodes'])
    reviews_df = pd.DataFrame(data['reviews'])

with open('dataset/cmu.json', 'r') as file:
    book_data = json.load(file)

with open('dataset/netflix.json', 'r') as file:
    show_data = json.load(file)
app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    vectorizer = TfidfVectorizer(max_df=max_df,min_df=min_df,
    max_features=max_features,stop_words=stop_words, norm=norm)
    return vectorizer

def title_idx_maker(db, ind):
    title_to_idx = {}
    idx_to_title = {}

    for index, row in enumerate(db):
        title_to_idx[row[ind]] = index
        idx_to_title[index] = row[ind]
    return title_to_idx, idx_to_title
book_title_to_idx, book_idx_to_title = title_idx_maker(book_data, "book_title")
netflix_title_to_idx, netflix_idx_to_title = title_idx_maker(show_data, "title")

n_feats = 5000
doc_by_vocab = np.empty([len(book_data), n_feats])
tfidf_vec = build_vectorizer(n_feats, "english")
doc_by_vocab = tfidf_vec.fit_transform([row['plot_summary'] for index, row in enumerate(book_data)]).toarray()
index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names_out())}

query_by_vocab = np.empty([len(show_data), n_feats])

query_by_vocab = tfidf_vec.transform([row['description'] for index, row in enumerate(show_data)]).toarray()

def get_sim_book(netflix_title, book_mat):
  if netflix_title in netflix_title_to_idx:
    netflix_idx = netflix_title_to_idx[netflix_title]
    netflix_vec = query_by_vocab[netflix_idx].reshape(1,-1)

    similarities = cosine_similarity(netflix_vec, book_mat)

    return similarities


def book_sims_to_recs(book_sims, book_idx_to_title):
  sim_pairs = [(book_idx_to_title[i], sim) for i, sim in enumerate(book_sims[0])]
  top_5 = sorted(sim_pairs, key=lambda x: x[1], reverse = True)[:5]
  return top_5


def rec_books(netflix_title, book_mat, book_idx_to_title):
    similarities = get_sim_book(netflix_title, book_mat)

    top_5 = book_sims_to_recs(similarities, book_idx_to_title)
    matches = (book_data[top_5]).to_json(orient = 'records')

    '''print(f"The 5 most similar books to {netflix_title} are:")
    for index, (book_title, sim_score) in enumerate(top_5):
        print(f"\n{index+1}. {book_title}")'''
    return matches
    
@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    '''return json_search(text)'''
    return rec_books(text, doc_by_vocab, book_idx_to_title)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)