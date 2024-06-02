import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer,MinMaxScaler,KBinsDiscretizer
import re
from sklearn.cluster import KMeans

class MovieRatingEstimator:
    """Once fit, this class can be used to estimate the rating for an unseen movie for a given user present in the training data.
    """
    def __init__(self) -> None:
        self.user_data = None
        self.user_clusters = None
        self.cluster_movie_avg_ratings = None

    def generate_movie_data(self,movies: pd.DataFrame):
        """Generates a new DataFrame with movieId, title and the genres + discretized year of release as one-hot encoded features.

           Attributes
           ----------
           * movies: DataFrame with columns movieId, title and genres (pipe separated strings)
        """
        movies_cpy = movies.copy()
        movies_cpy['genres'] = movies_cpy['genres'].str.split('|')

        mlb = MultiLabelBinarizer()
        new_movies = pd.DataFrame(mlb.fit_transform(movies_cpy['genres']),columns=mlb.classes_,index=movies.index)
        new_movies['title'] = movies_cpy['title']
        new_movies['movieId'] = movies_cpy['movieId']


        pattern = re.compile(r'\((\d{4})\)')
        new_movies['year'] = new_movies['title'].apply(lambda x: int(pattern.findall(x).pop()) if pattern.findall(x) else -1)

        new_movies['year'] = KBinsDiscretizer(n_bins=5,strategy='kmeans',encode='ordinal').fit_transform(new_movies['year'].values.reshape(-1,1))
        new_movies = pd.get_dummies(new_movies,columns=['year'],dtype=int)

        return new_movies

    def collect_user_data(self,
                          ratings: pd.DataFrame,
                          movies: pd.DataFrame,
                          verbose: bool = False):
        """Collects the average rating for each genre for each user in the training data.

           Attributes
           ----------
           * ratings: DataFrame with columns userId, movieId and rating
           * movies: DataFrame with columns movieId, title and one-hot encoded features -> check generate_movie_data()
           * verbose: Toggle logging
        """
        self.global_mean = ratings.rating.mean()
        
        genres = movies.drop(columns=['title','movieId']).columns
        genre_sums = {x:0 for x in genres}
        genre_counts = {x:0 for x in genres}
        
        user_avgs = {x:dict() for x in ratings.userId.unique()}

        multiplier = 1000 # To avoid floating point errors
        
        if verbose: print('Collecting feature averages for each user...')
        for genre in genres:
            if verbose: print(f"\tFeature: {genre}")

            genre_movies = movies[movies[genre] == 1]
            genre_merge = ratings.merge(genre_movies[[genre,"movieId"]],on='movieId',how='inner')

            genre_counts[genre] = genre_merge[genre].sum(axis=0)

            genre_sums[genre] = (genre_merge['rating']*multiplier).sum(axis=0)

            genre_merge['rating'] = genre_merge['rating']*multiplier
            groupped_user = genre_merge[['userId','rating']].groupby('userId').mean()
            
            for userId in user_avgs.keys():
                if userId in groupped_user.index:
                    user_avgs[userId][genre] = groupped_user.loc[userId,'rating']/multiplier
                else:
                    user_avgs[userId][genre] = np.nan

        genre_avgs = {x:(genre_sums[x]/genre_counts[x])/1000 if genre_counts[x] > 0 else 0.5 for x in genres}

        self.user_data = pd.DataFrame(user_avgs).T.fillna(genre_avgs).fillna(self.global_mean)
        self.user_data['userId'] = user_avgs.keys()

        print('Done :)')

    def fit(self,
            ratings: pd.DataFrame,
            movies: pd.DataFrame,
            n_clusters:int = 10,
            force_recollect: bool = False,
            verbose: bool = False):
        """Fits the model to the training data.

           Attributes
           ----------
           * ratings: DataFrame with columns userId, movieId and rating
           * movies: DataFrame with columns movieId, title and genres (pipe separated strings)
           * n_clusters: Number of clusters to use for KMeans clustering
           * force_recollect: Toggle data recalculation
           * verbose: Toggle logging
        """
        self.scaler = MinMaxScaler()

        rate_cpy = ratings.copy()
        rate_cpy['rating'] = self.scaler.fit_transform(rate_cpy['rating'].values.reshape(-1,1))

        

        if self.user_data is None or force_recollect:
            self.collect_user_data(rate_cpy,self.generate_movie_data(movies),verbose=verbose)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init=10)
        
        cluster_df = pd.DataFrame(self.user_data.userId)
        cluster_df['cluster'] = kmeans.fit_predict(self.user_data.drop('userId',axis=1))

        ids = ratings.userId.unique()
        cluster_df = cluster_df[cluster_df.userId.isin(ids)]

        self.user_clusters = cluster_df.set_index('userId').to_dict()['cluster']
        
        
        rate_cpy = rate_cpy.merge(cluster_df,on='userId')

        self.cluster_movie_avg_ratings = dict()
        for x in range(n_clusters):
            self.cluster_movie_avg_ratings[x] = rate_cpy[rate_cpy.cluster == x].drop(columns=['cluster','userId']).groupby('movieId').mean()

    def get_rating_estimate(self,userId:int, movieId:int):
        """Estimates the movie rating for a given user-movie pair. (Normalized)

           Attributes
           ----------
           * userId: User ID
           * movieId: Movie ID
        """
        if self.cluster_movie_avg_ratings is None:
            raise Exception('Use fit')
        
        if userId not in self.user_clusters:
            return self.global_mean
        
        user_cluster = self.user_clusters[userId]
        cluster_movie_avg_ratings = self.cluster_movie_avg_ratings[user_cluster]

        try:
            return cluster_movie_avg_ratings.loc[movieId].rating
        except:
            return self.global_mean
        
    def get_rating_estimate_in_scale(self,userId, movieId):
        """Estimates the movie rating for a given user-movie pair. (Training data scale)

           Attributes
           ----------
           * userId: User ID
           * movieId: Movie ID
        """
        return self.scaler.inverse_transform([[self.get_rating_estimate(userId,movieId)]])[0][0]