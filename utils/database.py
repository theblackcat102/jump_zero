from datetime import datetime
from utils.settings import IP
import os, sys, glob
import pickle
# myclient = pymongo.MongoClient("mongodb://{}:27017/".format(IP))
# mydb = myclient["alpha_jump"]

class Collection:
    def __init__(self, code_version, model_version):
        code_path = os.path.join('./data/', 'code_'+code_version)
        os.makedirs(code_path, exist_ok=True)
        model_path = os.path.join(code_path, 'model_'+model_version)
        os.makedirs(model_path, exist_ok=True)
        self.model_path = model_path
    
    def add_batch(self, games):
        ids = []
        for game in games:
            filename = game['time'].strftime("%m_%d_%Y_%H_%M_%S_%f")+'.pkl'
            file_path = os.path.join(self.model_path, filename)
            with open(file_path, 'wb') as f:
                pickle.dump(game, f)
            ids.append(file_path)
        return ids
    
    def add_game(self, game):
        filename = game['time'].strftime("%m_%d_%Y_%H_%M_%S_%f")+'.pkl'
        file_path = os.path.join(self.model_path, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(game, f)
        return file_path

    def get_all(self):
        file_path = os.path.join(self.model_path, '*.pkl')
        games = []
        filenames = [ (filename, datetime.strptime(os.path.basename(filename), "%m_%d_%Y_%H_%M_%S_%f.pkl")) for filename in glob.glob(file_path)]
        filenames = sorted(filenames, key=lambda x: x[1], reverse=True)
        for filename, _ in filenames:
            with open(filename, 'rb') as f:
                game = pickle.load(f)
                games.append(game)
        return games
