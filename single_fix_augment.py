import tensorflow as tf
from keras.layers import *
import warnings,os,argparse,datetime
import lr
import numpy as np
import matplotlib.pyplot as plt
from call_model import get_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.callbacks import CSVLogger 
import albumentations as A
import tensorflow_addons as tfa
from data_loader_single import MyGenerator_fix_augment
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning) #Ignore DeprecationWarning

def train(model, validationNumber, batchsize, filter_depth, input_shape, params):
    # Define a model      
    net = get_model(model=model, input_shape=input_shape, d=filter_depth)
    # # 新添
    # if model == 'LLNet':
    #     net = LLNet_modified(input_shape)
    # elif model == 'Resnext':
    #     net = Resnext(input_shape, 17)
    # elif model == 'sen2LCZ_drop_core':
    #     net = sen2LCZ_drop_core(input_shape, depth=5)
    # elif model == 'RSNNet':
    #     net = RSNNet(input_shape, 17)
    # elif model == 'LCZNet':
    #     net = LCZNet(input_shape)
    # elif model == 'CNN_1':
    #     net = CNN_1((32,32,10))
    # elif model == 'CNN_2':
    #     net = CNN_2((32,32,10))
    net.summary()
    net.compile(optimizer = tf.keras.optimizers.Nadam(), loss = 'categorical_crossentropy', 
                metrics=['accuracy','Precision','Recall', 
                         tfa.metrics.F1Score(num_classes=17,average='weighted',name='f1_weighted')])   
    # Augmentation
    augmentations = A.Compose([A.HorizontalFlip(p=0.5),A.Rotate(limit=(-90, 90)),A.VerticalFlip(p=0.5)])   
    
    # Hyperparameters
    lr_sched = lr.step_decay_schedule(initial_lr=params['initial_lr'], decay_factor=params['decay_factor'], step_size=params['step_size'])
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = params['patience'], verbose=1)    
    tb_name = model+'_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    epochs=params['epoch'] 
    
    # Record Hyperparameters
    txt_logger="./log/"+tb_name+".csv"
    with open(txt_logger, 'w') as file:
        for param, value in params.items():
            file.write(f'{param}: {value}\n')
    
    # Callbacks
    csv_logger = CSVLogger("./log/"+tb_name+".csv", append=True)
    modelbest = './model_weights/'+tb_name+'.hdf5'
    checkpoint = ModelCheckpoint(filepath=modelbest, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    # Train    
    history = net.fit_generator(MyGenerator_fix_augment('train',batch_size=batchsize, augmentations=augmentations, shuffle=True),
                    workers=1,          
                    validation_data = MyGenerator_fix_augment('validation',batch_size=batchsize, augmentations=augmentations, shuffle=True),
                    validation_steps = validationNumber//batchsize,
                    epochs=epochs,
                    max_queue_size=100,
                    callbacks=[early_stopping, lr_sched, checkpoint, csv_logger])   

    # Summarize history for accuracy
    print(history.history['accuracy'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.savefig('./fig/'+tb_name+'.png', dpi = 300)

def checknum():
    with open('./patches_split/partition_random.npz', 'rb') as f:
        loaded_indexes = pickle.load(f)
    
    train_indexes = loaded_indexes['train'] 
    validation_indexes = loaded_indexes['validation']
    return len(train_indexes),len(validation_indexes)

if __name__ == '__main__':
    # Define some arguments
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model', type=str, default='resnet11_3D', help='Deep learning Model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_dims', type=tuple, default=(32,32,10), help='Shape of images')
    parser.add_argument('--initial_lr', type=float, default=0.002, help='initial learning rate')
    parser.add_argument('--decay_factor', type=float, default=0.5, help='decay factor rate')
    parser.add_argument('--step_size', type=int, default=5, help='step size')
    parser.add_argument('--patience', type=int, default=15, help='patience')
    parser.add_argument('--epoch', type=int, default=100, help='epoch')

    
    args = parser.parse_args()
    model_name = args.model
    input_shape=args.img_dims
    batchsize=args.batch_size

    params = {
    'initial_lr': args.initial_lr,
    'decay_factor': args.decay_factor,
    'step_size': args.step_size,
    'patience': args.patience,
    'epoch':args.epoch
    }

    _trainNumber,validationNumber = checknum()

    train(model_name, validationNumber, batchsize=batchsize, filter_depth=16, input_shape=input_shape, params=params)