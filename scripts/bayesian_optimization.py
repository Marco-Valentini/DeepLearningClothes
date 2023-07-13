# TODO vedi link antonio e completa qui
# https://maelfabien.github.io/machinelearning/HyperOpt/#the-data
# https://github.com/hyperopt/hyperopt/wiki/FMin

# import model to optimize and the function to optimize
### HyperOpt Parameter Tuning
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
### import model to optimize and the function to optimize
from BERT_architecture import umBERT2
from utility import umBERT2_trainer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

