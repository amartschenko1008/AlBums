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
netflix_db.to_json('netflix.json', orient ='records')

'''parts = ['id','title','type','description','release_year','age_certification','runtime','genres','production_countries','seasons','imdb_id','imdb_score','imdb_votes','tmdb_popularity','tmdb_score']

with open(filename) as fh:
    for line in fh:
        desc = line.strip().split(",")
        print(desc[14])
        dict2 = {}
        i = 0
        while i < len(parts):
            dict2[parts[i]] = desc[i]
            i += 1
        books[desc[3]] = dict2
'''
'''out_file = open("test.json", "w")
json.dump(books, out_file, indent = 4)
out_file.close()'''