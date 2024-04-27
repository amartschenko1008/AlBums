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
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import spacy


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


# Code for spacy lemmatization from this link: https://jonathansoma.com/lede/image-and-sound/text-analysis/text-analysis-word-counting-lemmatizing-and-tf-idf/
spacy_lemma = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatize(text):
    doc = spacy_lemma(text)
    # Turn it into tokens, ignoring the punctuation
    tokens = []
    for token in doc:
        if not token.is_punct:
            if token.pos_ != "PRON":
                tokens.append(token.lemma_)
            else:
                tokens.append(token.orth_)

    """
    tokens = [
        token.lemma_ if token.pos_ != "PRON" else token.orth_
        for token in doc
        if not token.is_punct
    ]
    """
    return tokens


def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm="l2"):
    vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        stop_words=stop_words,
        norm=norm,
        tokenizer=lemmatize,
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

n_feats = 20000
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

n_components = 100

svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
trained_svd = svd.fit(doc_by_vocab)

svd_books = trained_svd.transform(doc_by_vocab)
svd_netflix = trained_svd.transform(query_by_vocab)

zero_to_1_scaler = MinMaxScaler().fit(svd_books)

normalized_book_mat = zero_to_1_scaler.transform(svd_books)
normalized_netflix_mat = zero_to_1_scaler.transform(svd_netflix)

compressed_terms = trained_svd.components_.T


def get_sim_book(netflix_titles, book_mat):
    # if input is empty, replace input with a string
    if not netflix_titles:
        netflix_titles = [""]
    lower_titles = [a.lower() for a in netflix_titles]

    # These vectors are used to generate charts.
    netflix_vectors_for_graphs = []
    for lower_title in lower_titles:
        if lower_title in netflix_title_to_idx:
            netflix_idx = netflix_title_to_idx[lower_title]
            netflix_vec = normalized_netflix_mat[netflix_idx]
            netflix_vectors_for_graphs.append(netflix_vec)
        else:
            # If title not in DB, search for those terms instead
            tf_idf_query = scipy.sparse.csr_matrix.toarray(
                tfidf_vec.transform([lower_title])
            )
            query_vec = trained_svd.transform(tf_idf_query).reshape(
                n_components,
            )

            normalized_query_vec = zero_to_1_scaler.transform(query_vec.reshape(1, -1))
            normalized_query_vec = normalized_query_vec.reshape(
                n_components,
            )

            netflix_vectors_for_graphs.append(normalized_query_vec)

    # These vectors are used for calculating similarity
    netflix_vectors = []
    for lower_title in lower_titles:
        if lower_title == "":
            pass
        elif lower_title in netflix_title_to_idx:
            netflix_idx = netflix_title_to_idx[lower_title]
            netflix_vec = normalized_netflix_mat[netflix_idx]
            netflix_vectors.append(netflix_vec)
        else:
            # If title not in DB, search for those terms instead
            tf_idf_query = scipy.sparse.csr_matrix.toarray(
                tfidf_vec.transform([lower_title])
            )
            query_vec = trained_svd.transform(tf_idf_query).reshape(
                n_components,
            )

            normalized_query_vec = zero_to_1_scaler.transform(query_vec.reshape(1, -1))
            normalized_query_vec = normalized_query_vec.reshape(
                n_components,
            )

            netflix_vectors.append(normalized_query_vec)

    # print(f"\n{netflix_vectors=}\n")

    avg_vector = np.mean(netflix_vectors, axis=0).reshape(1, -1)
    similarities = cosine_similarity(avg_vector, book_mat)

    query_vector_list = []
    for idx, title in enumerate(lower_titles):
        if title == "":
            query_vector_list.append((title, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        else:
            """scaled_query_vec = zero_to_1_scaler.transform(
                netflix_vectors[idx].reshape(1, -1)
            ).tolist()[0][:10]
            query_vector_list.append((title, scaled_query_vec))
            """
            query_vector_list.append(
                (title, netflix_vectors_for_graphs[idx].tolist()[:10])
            )
    return similarities, query_vector_list


def book_sims_to_recs(book_sims, book_idx_to_title, book_mat, query_vector_list):
    if np.array_equal(book_sims, book_mat):
        return [("This title is not in our database.", None)]
    else:
        # sim_pairs = [(book_idx_to_title[i], sim) for i, sim in enumerate(book_sims[0])]
        title_sim_vec = [
            (book_idx_to_title[i], sim, normalized_book_mat[i].tolist())
            for i, sim in enumerate(book_sims[0])
        ]
        # top_5 = sorted(sim_pairs, key=lambda x: x[1], reverse=True)[:5]
        top_5 = sorted(title_sim_vec, key=lambda x: x[1], reverse=True)[:5]
        # Title, sim, summary, vector, ranking, query_vector_list
        top_5 = [
            (
                t[0],
                t[1],
                book_data[book_title_to_idx[t[0].lower()]]["plot_summary"],
                t[2][:10],
                idx,
                query_vector_list,
            )
            for idx, t in enumerate(top_5)
        ]

        return top_5


def rec_books(netflix_title, book_mat, book_idx_to_title):
    assert book_mat is not None and book_idx_to_title is not None
    similarities, query_vector_list = get_sim_book(netflix_title, book_mat)
    top_5 = book_sims_to_recs(
        similarities, book_idx_to_title, book_mat, query_vector_list
    )
    """df= pd.DataFrame(dict( r = similarities.tolist(), theta = ["secrecy", "destruction", "contemporary", "government", "family", "magic", "morality", "travel"]))
    fig = px.line_polar(df, r='r', theta = 'theta', line_closed=True)
    fig.update_traces(fill='toself')
    fig.show()"""
    # top_5_list = [tup[0] for tup in top_5]
    top_5_list = [tup for tup in top_5]
    # print(f"{top_5_list=}")
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
    # print(f"{text1=}, {text2=}, {text3=}")
    titles = [text1, text2, text3]
    titles = [a for a in titles if a != None]
    """return json_search(text)"""
    return rec_books(titles, normalized_book_mat, book_idx_to_title)


if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
