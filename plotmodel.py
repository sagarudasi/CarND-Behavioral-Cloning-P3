from keras.utils import plot_model
from keras.models import load_model
import sys

model = load_model(sys.argv[1])

plot_model(model, to_file='model.png')