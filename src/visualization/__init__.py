import numpy as np
import pandas as pd
import os
import pickle
from flask import Flask
from tensorflow import keras


# directory with models
cwd = os.getcwd()
saved_models_path = os.path.join(cwd, 'models/')

# load Deep Neural Network model
NN_MODEL_NAME = 'speakers_classification.hdf5'
nn_model = keras.models.load_model(saved_models_path + NN_MODEL_NAME)

# load Gaussian Mixture model
gmm_files = [os.path.join('models/gmm/', fname) for fname in os.listdir('models/gmm/') if fname.endswith('.sav')]

# load the Gaussian gender models
gmm_models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]

# load classes file, order of speakers is preserved
classes = np.load(saved_models_path + 'classes.npy', allow_pickle=True)

# load metadata file for mapping id to person's name
csv_path = os.path.join(cwd, 'data/')
metadata = pd.read_csv(csv_path + 'vox1_meta.csv', sep='\t')
metadata = metadata.drop(['Gender',	'Nationality',	'Set'], axis=1)
metadata = metadata[metadata['VoxCeleb1 ID'].isin(classes)]


def create_app():
    # name of the file initialize flask
    app = Flask(__name__)

    print(nn_model.summary())
    print(f'Classes are: {classes}')
    print(f'Current speakers shape: {metadata.shape}')

    # register views/routes
    from .views import views

    app.register_blueprint(views, url_prefix='/')

    return app
