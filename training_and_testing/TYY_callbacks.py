import keras
from sklearn.metrics import roc_auc_score
import sys
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from keras import backend as K


class DecayLearningRate(keras.callbacks.Callback):
	def __init__(self, startEpoch):
		self.startEpoch = startEpoch

	def on_train_begin(self, logs={}):
		return
	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		
		if epoch in self.startEpoch:
			if epoch == 0:
				ratio = 1
			else:
				ratio = 0.1
			LR = K.get_value(self.model.optimizer.lr)
			K.set_value(self.model.optimizer.lr,LR*ratio)
		return

	def on_epoch_end(self, epoch, logs={}):
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
