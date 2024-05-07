"""
This script implements a function to conduct predictions and fuse the two stream: Sentinel-2 and Google Earth.

Author: ctrlovefly
Date: January 21, 2024

"""
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
import numpy as np
from call_model import get_model
import tensorflow
import tensorflow_addons as tfa
from data_loader_single import MyGenerator_fix_augment,MyDataset_simplified_test
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.data import DisjointLoader

def cohen_kappa(confusion_matrix):
        total_samples = confusion_matrix.sum()
        num_classes = confusion_matrix.shape[0]
        # po
        po = sum(confusion_matrix[i, i] for i in range(num_classes)) / total_samples
        # pe
        pe = sum((sum(confusion_matrix[:, i]) * sum(confusion_matrix[i, :])) / (total_samples ** 2) for i in range(num_classes))
        # Cohen's Kappa
        kappa = (po - pe) / (1 - pe) 
        return kappa
    
def overall_accuracy_top_classes(confusion_matrix, num_classes=10):
    # the first 10 classes
    total_samples = sum(confusion_matrix[i, j] for i in range(num_classes) for j in range(len(confusion_matrix)))      
    correct_predictions = sum(confusion_matrix[i, i] for i in range(num_classes))
    overall_accuracy = correct_predictions / total_samples    
    return overall_accuracy

def overall_accuracy_last_classes(confusion_matrix, num_classes=7):
    # the last 7 classes
    total_samples_last_classes = sum(confusion_matrix[i, j] for i in range(len(confusion_matrix)) for j in range(len(confusion_matrix)) if i >= (len(confusion_matrix) - num_classes))
    correct_predictions_last_classes = sum(confusion_matrix[i, j] for i in range(len(confusion_matrix)) for j in range(len(confusion_matrix)) if i >= (len(confusion_matrix) - num_classes) and i == j)
    overall_accuracy_last_classes = correct_predictions_last_classes / total_samples_last_classes
    return overall_accuracy_last_classes

def accuracy_per_class(conf_matrix):
    # OA per class
    num_classes = len(conf_matrix)
    accuracies_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        true_positive_i = conf_matrix[i, i]
        false_positive_i = np.sum(conf_matrix[:, i]) - true_positive_i

        if (true_positive_i + false_positive_i) > 0:
            accuracies_per_class[i] = true_positive_i / (true_positive_i + false_positive_i)

    print("Accuracies per class:", accuracies_per_class)
    return accuracies_per_class

Alpha_list=[0.6]# Change weighting alpha
for Alpha in Alpha_list:   
    # Sentinel-2 stream
    weights_files='resnet11_3D_20240507-100329'# Change to your weight files
    model_path="./model_weights/"+weights_files+".hdf5"
    model = get_model(model="resnet11_3D", input_shape=[32,32,10], d=16)
    model.compile(optimizer = tensorflow.keras.optimizers.Nadam(), loss = 'categorical_crossentropy', 
                metrics=['accuracy','Precision','Recall', 
                        tfa.metrics.F1Score(num_classes=17,average='weighted',name='f1_weighted')]) 
    model.load_weights(model_path)
    # Google Earth stream
    weights_files_2='gcn_20240507-093556'# Change to your weight files
    model_path2="./model_weights/"+weights_files_2+".hdf5"    
    # model2=Net()
    model2 = get_model(model="gnn", input_shape=[32,32,10], d=16)
    model2.compile(optimizer = tensorflow.keras.optimizers.Nadam(), loss = 'categorical_crossentropy', 
                metrics=['accuracy','Precision','Recall', 
                            tfa.metrics.F1Score(num_classes=17,average='weighted',name='f1_weighted')]) 
    model2.load_weights(model_path2)

    batchsize = 64
    # Load the testing Sentinel-2 images and predict them 
    generator= MyGenerator_fix_augment('test',batch_size=batchsize,shuffle=False)
    predictions_list = model.predict(generator, verbose=1)
    
    # Load the testing corresponding Google Earth images and predict them
    test_labels=[] 
    data1=MyDataset_simplified_test(transforms=NormalizeAdj())
    loader_tr1 = DisjointLoader(data1, batch_size=batchsize, epochs=1, shuffle=False)
    predictions_list1=[]
    idx=0

    for batch in loader_tr1:
        idx=idx+1
        inputs, target = batch
        predictions = model2.predict(inputs, batch_size = inputs[-1].shape[0])
        predictions_list1.append(predictions)
        test_labels.append(target)
    predictions_list1 = np.vstack(predictions_list1)
    test_labels = np.vstack(test_labels)
    # Weighted Fusion
    predictions_sum =Alpha*predictions_list+(1-Alpha)*predictions_list1 
    # Generate Confusion Matrix
    predicted_classes = np.argmax(predictions_sum, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:")
    print(conf_matrix)
    # Calculate several metrics based on the Confusion Matrix
    oanb=overall_accuracy_last_classes(conf_matrix)
    oab=overall_accuracy_top_classes(conf_matrix)
    kappa=cohen_kappa(conf_matrix)
    accuracy = accuracy_score(true_classes, predicted_classes)
    avg_f1 = f1_score(true_classes, predicted_classes,average='weighted')
    accperclass=accuracy_per_class(conf_matrix)
    print("oab:", oab)
    print("oanb:", oanb)
    print("kappa:", kappa)
    print("Accuracy:", accuracy)
    print("Average F1 score:", avg_f1)

 