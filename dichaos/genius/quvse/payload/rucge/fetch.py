import asyncio, pdb, random
import pandas as pd


class Fetch(object):

    def read_pd(self, path_file):
        data = pd.read_csv(path_file, index_col=0)
        return data

    def fetch_random_features(self, path_file, k=50):
        df = self.read_pd(path_file=path_file)
        df1 = df[['Field', 'Description']].to_dict(orient='records')
        random_data = random.sample(df1, k)
        return dict(
            zip([k['Field'] for k in random_data],
                [k['Description'] for k in random_data]))
