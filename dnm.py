from recommender.cf_model import CollaborativeFilteringModel
model = CollaborativeFilteringModel(data_path='data/ml-latest-small')
model.load_data()
model.build_matrix()
results = model.get_popular_movies(n=10, min_ratings=50)
for r in results:
    print(r['title'], r['avg_rating'], r['num_ratings'])