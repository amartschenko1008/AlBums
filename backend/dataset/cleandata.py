import json
import pandas as pd
filename = 'titles.csv'

netflix_db = pd.read_csv("titles.csv")
netflix_db = netflix_db.drop_duplicates(subset="title", ignore_index=True)
netflix_db = netflix_db.fillna("")

netflix_title_to_idx = {}
netflix_idx_to_title = {}
for index, row in netflix_db.iterrows():
  netflix_title_to_idx[row['title']] = index
  netflix_idx_to_title[index] = row['title']
i = netflix_db[(netflix_db['type'] == 'MOVIE')]
netflix_db = netflix_db[(netflix_db['type'] == 'SHOW')]
netflix_db = netflix_db.drop(netflix_db.columns[[0]], axis = 1)
netflix_db.to_json('netflix.json', orient ='records',  indent = 4)

column_names = ["wikipedia_id", "freebase_id", "book_title","book_author","book_pub_date",
                "book_genres","plot_summary"]
cmu_db = pd.read_table("booksummaries.txt", names=column_names)
cmu_db = cmu_db.drop_duplicates(subset="book_title", ignore_index= True)
cmu_db = cmu_db.drop(cmu_db.columns[[0, 1]], axis = 1)
cmu_db.to_json('cmu.json', orient ='records', indent = 4)
