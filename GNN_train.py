"""
This script implements the training of the Google Earth stream.

Author: ctrlovefly
Date: January 21, 2024

"""
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from spektral.data import DisjointLoader
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj
from data_loader_single import MyDataset_simplified,MyDataset_simplified_val
from keras.callbacks import EarlyStopping
import lr
import datetime
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,ModelCheckpoint

################################################################################
# # Data
################################################################################
data=MyDataset_simplified(transforms=NormalizeAdj())
data_val=MyDataset_simplified_val(transforms=NormalizeAdj())  
# ################################################################################
# # Config
# ################################################################################
initial_lr=0.01 # Learning rate
epochs = 100  # Number of training epochs
es_patience = 20  # Patience for early stopping
batch_size =32  # Batch size
# ################################################################################
# # Loader
# ################################################################################
loader_tr = DisjointLoader(data, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_val, batch_size=batch_size)
# ################################################################################
# # Network
# ################################################################################
class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GCSConv(32, activation="relu")
        self.conv2 = GCSConv(32, activation="relu")
        self.conv3 = GCSConv(32, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(data.n_labels, activation="softmax")

    def call(self, inputs):
        x, a, i = inputs
        print(i)
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        print(x.shape)
        output = self.global_pool([x, i])
        print(output.shape)
        output = self.dense(output)
        
        return output
# ################################################################################
# # Net instance
# ################################################################################
model = Net()
model.compile(
    optimizer=tf.keras.optimizers.Nadam(),
    loss=CategoricalCrossentropy(reduction="sum"),
    metrics=['accuracy','Precision','Recall',
                        tfa.metrics.F1Score(num_classes=17,average='weighted',name='f1_weighted')])   
# Summary
dummy_x_input = tf.keras.Input(shape=(5,),dtype=tf.float64)
dummy_a_input = tf.keras.Input(shape=(None,), sparse=True)
dummy_i_input = tf.keras.Input(shape=(),dtype=tf.int64)
_ = model((dummy_x_input,dummy_a_input,dummy_i_input))
model.summary()
# ################################################################################
# # train
# ################################################################################
lr_sched = lr.step_decay_schedule(initial_lr=initial_lr, decay_factor=1, step_size=5)
# Callbacks
model_name='gcn'
tb_name = model_name+'_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelbest = './model_weights/'+tb_name+'.hdf5'
checkpoint = ModelCheckpoint(filepath=modelbest, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
# Train
print( loader_tr.load())
history = model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(monitor='val_loss',patience=es_patience,verbose=1), lr_sched, checkpoint],
)
# Summarize history for accuracy
print(history.history['accuracy'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig('./fig/'+tb_name+'.png', dpi = 300)

