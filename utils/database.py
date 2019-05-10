import pymongo
import datetime as datetime
from utils.settings import IP

myclient = pymongo.MongoClient("mongodb://{}:27017/".format(IP))
mydb = myclient["alpha_jump"]

class Collection:

    def __init__(self, code_version, model_version):
        self.col = mydb['code_'+code_version+'model_'+model_version]
    
    def add_batch(self, games):
        ids = self.col.insert_many(games)
        return ids

    def add_game(self, game):
        return self.col.insert_one(game)
    
    def get_all(self):
        return self.col.find()

if __name__ == "__main__":
    import numpy as np
    collection = Collection('test', 'v0.1')
    collection.add_game({
        'test': [np.zeros((8,8)).tolist()]
    })
    for row in collection.get_all():
        print(row)