import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import scipy


# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, "init.json")

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, "r") as file:
    data = json.load(file)
    episodes_df = pd.DataFrame(data["episodes"])
    reviews_df = pd.DataFrame(data["reviews"])

with open("dataset/cmu.json", "r") as file:
    book_data = json.load(file)
    book_df = pd.DataFrame(book_data)
    book_df = book_df.drop_duplicates(subset="book_title", ignore_index=True)

with open("dataset/netflix.json", "r") as file:
    show_data = json.load(file)
    show_df = pd.DataFrame(show_data)
    show_df = show_df.drop_duplicates(subset="title", ignore_index=True)

app = Flask(__name__)
CORS(app)


# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(
        episodes_df, reviews_df, left_on="id", right_on="id", how="inner"
    )
    matches = merged_df[merged_df["title"].str.lower().str.contains(query.lower())]
    matches_filtered = matches[["title", "descr", "imdb_rating"]]
    matches_filtered_json = matches_filtered.to_json(orient="records")
    return matches_filtered_json


def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm="l2"):
    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        stop_words=stop_words,
        norm=norm,
    )
    return vectorizer


def title_idx_maker(db, ind):
    title_to_idx = {}
    idx_to_title = {}

    for index, row in enumerate(db):
        title_to_idx[row[ind].lower()] = index
        idx_to_title[index] = row[ind]
    return title_to_idx, idx_to_title


book_title_to_idx, book_idx_to_title = title_idx_maker(book_data, "book_title")
netflix_title_to_idx, netflix_idx_to_title = title_idx_maker(show_data, "title")

n_feats = 10000
doc_by_vocab = np.empty([len(book_data), n_feats])
tfidf_vec = build_vectorizer(n_feats, "english")
doc_by_vocab = tfidf_vec.fit_transform(
    [row["plot_summary"] for index, row in enumerate(book_data)]
).toarray()
index_to_vocab = {i: v for i, v in enumerate(tfidf_vec.get_feature_names_out())}

query_by_vocab = np.empty([len(show_data), n_feats])

query_by_vocab = tfidf_vec.transform(
    [row["description"] for index, row in enumerate(show_data)]
).toarray()

n_components = 10

svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
trained_svd = svd.fit(doc_by_vocab)

svd_books = trained_svd.transform(doc_by_vocab)
svd_netflix = trained_svd.transform(query_by_vocab)

compressed_terms = trained_svd.components_.T


def get_sim_book(netflix_titles, book_mat):
    # if input is empty, replace input with a string
    if not netflix_titles:
        netflix_titles = [""]
    lower_titles = [a.lower() for a in netflix_titles]
    netflix_vectors = []
    for lower_title in lower_titles:
        if lower_title in netflix_title_to_idx:
            netflix_idx = netflix_title_to_idx[lower_title]
            netflix_vec = svd_netflix[netflix_idx]
            netflix_vectors.append(netflix_vec)
        else:
            # If title not in DB, search for those terms instead
            tf_idf_query = scipy.sparse.csr_matrix.toarray(
                tfidf_vec.transform([lower_title])
            )
            query_vec = trained_svd.transform(tf_idf_query).reshape(
                n_components,
            )
            print(f"{query_vec.shape=}")
            netflix_vectors.append(query_vec)
    avg_vector = np.mean(netflix_vectors, axis=0).reshape(1, -1)
    similarities = cosine_similarity(avg_vector, book_mat)
    return similarities


def book_sims_to_recs(book_sims, book_idx_to_title, book_mat):
    if np.array_equal(book_sims, book_mat):
        return [("This title is not in our database.", None)]
    else:
        sim_pairs = [(book_idx_to_title[i], sim) for i, sim in enumerate(book_sims[0])]
        top_5 = sorted(sim_pairs, key=lambda x: x[1], reverse=True)[:5]
        top_5 = [(t[0],t[1],book_data[book_title_to_idx[t[0].lower()]]["plot_summary"]) for t in top_5]
        return top_5


def rec_books(netflix_title, book_mat, book_idx_to_title):
    assert book_mat is not None and book_idx_to_title is not None
    similarities = get_sim_book(netflix_title, book_mat)

    top_5 = book_sims_to_recs(similarities, book_idx_to_title, book_mat)
    # top_5_list = [tup[0] for tup in top_5]
    top_5_list = [tup for tup in top_5]
    print(f"{top_5_list=}")
    # matches = (book_data[top_5]).to_json(orient="records")
    # print(top_5)

    matches = json.dumps(top_5_list)

    """print(f"The 5 most similar books to {netflix_title} are:")
    for index, (book_title, sim_score) in enumerate(top_5):
        print(f"\n{index+1}. {book_title}")"""
    return matches


@app.route("/")
def home():
    return render_template("base.html", title="sample html")


@app.route("/episodes")
def episodes_search():
    text1 = request.args.get("title1")
    text2 = request.args.get("title2")
    text3 = request.args.get("title3")
    print(f"{text1=}, {text2=}, {text3=}")
    titles = [text1, text2, text3]
    titles = [a for a in titles if a != None]
    """return json_search(text)"""
    return rec_books(titles, svd_books, book_idx_to_title)


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
