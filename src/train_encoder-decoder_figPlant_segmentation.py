from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from random import shuffle
import matplotlib.pyplot as plt
import os, shutil
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imsave
import time
import math
from itertools import permutations

def obtener_sets():
    
    dict_images_RGB = dict()
    dict_images_GT = dict()
    
    dict_lista_RGB = dict()
    dict_lista_GT = dict()
    
    DJI_0010_A_carpeta = os.path.join(original_set, 'DJI_0010_A')
    DJI_0010_B_carpeta = os.path.join(original_set, 'DJI_0010_B') 
    DJI_0018_A_carpeta = os.path.join(original_set, 'DJI_0018_A')
    DJI_0036_A_carpeta = os.path.join(original_set, 'DJI_0036_A')
    DJI_0043_A_carpeta = os.path.join(original_set, 'DJI_0043_A')
    DJI_0051_A_carpeta = os.path.join(original_set, 'DJI_0051_A')
    DJI_0075_A_carpeta = os.path.join(original_set, 'DJI_0075_A') 
    DJI_0083_A_carpeta = os.path.join(original_set, 'DJI_0083_A')
    DJI_0098_A_carpeta = os.path.join(original_set, 'DJI_0098_A')
    DJI_0101_A_carpeta = os.path.join(original_set, 'DJI_0101_A')

    DJI_0010_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0010_A_mask_0')
    DJI_0010_B_mask_0_carpeta = os.path.join(original_set, 'DJI_0010_B_mask_0') 
    DJI_0018_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0018_A_mask_0')
    DJI_0036_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0036_A_mask_0') 
    DJI_0043_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0043_A_mask_0')
    DJI_0051_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0051_A_mask_0')
    DJI_0075_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0075_A_mask_0') 
    DJI_0083_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0083_A_mask_0')
    DJI_0098_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0098_A_mask_0') 
    DJI_0101_A_mask_0_carpeta = os.path.join(original_set, 'DJI_0101_A_mask_0')

    lista_DJI_0010_A = os.listdir( DJI_0010_A_carpeta )
    lista_DJI_0010_B = os.listdir( DJI_0010_B_carpeta ) 
    lista_DJI_0018_A = os.listdir( DJI_0018_A_carpeta )
    lista_DJI_0036_A = os.listdir( DJI_0036_A_carpeta )
    lista_DJI_0043_A = os.listdir( DJI_0043_A_carpeta )
    lista_DJI_0051_A = os.listdir( DJI_0051_A_carpeta )
    lista_DJI_0075_A = os.listdir( DJI_0075_A_carpeta ) 
    lista_DJI_0083_A = os.listdir( DJI_0083_A_carpeta )
    lista_DJI_0098_A = os.listdir( DJI_0098_A_carpeta )
    lista_DJI_0101_A = os.listdir( DJI_0101_A_carpeta )

    lista_DJI_0010_A.sort()
    lista_DJI_0010_B.sort()
    lista_DJI_0018_A.sort()
    lista_DJI_0036_A.sort()
    lista_DJI_0043_A.sort()
    lista_DJI_0051_A.sort()
    lista_DJI_0075_A.sort()
    lista_DJI_0083_A.sort()
    lista_DJI_0098_A.sort()
    lista_DJI_0101_A.sort()

    lista_DJI_0010_A_mask_0 = os.listdir( DJI_0010_A_mask_0_carpeta )
    lista_DJI_0010_B_mask_0 = os.listdir( DJI_0010_B_mask_0_carpeta ) 
    lista_DJI_0018_A_mask_0 = os.listdir( DJI_0018_A_mask_0_carpeta )
    lista_DJI_0036_A_mask_0 = os.listdir( DJI_0036_A_mask_0_carpeta ) 
    lista_DJI_0043_A_mask_0 = os.listdir( DJI_0043_A_mask_0_carpeta )
    lista_DJI_0051_A_mask_0 = os.listdir( DJI_0051_A_mask_0_carpeta )
    lista_DJI_0075_A_mask_0 = os.listdir( DJI_0075_A_mask_0_carpeta ) 
    lista_DJI_0083_A_mask_0 = os.listdir( DJI_0083_A_mask_0_carpeta )
    lista_DJI_0098_A_mask_0 = os.listdir( DJI_0098_A_mask_0_carpeta ) 
    lista_DJI_0101_A_mask_0 = os.listdir( DJI_0101_A_mask_0_carpeta )

    lista_DJI_0010_A_mask_0.sort()
    lista_DJI_0010_B_mask_0.sort()
    lista_DJI_0018_A_mask_0.sort()
    lista_DJI_0036_A_mask_0.sort()
    lista_DJI_0043_A_mask_0.sort()
    lista_DJI_0051_A_mask_0.sort()
    lista_DJI_0075_A_mask_0.sort()
    lista_DJI_0083_A_mask_0.sort()
    lista_DJI_0098_A_mask_0.sort()
    lista_DJI_0101_A_mask_0.sort()

    
    ruta_DJI_0010_A = []
    ruta_DJI_0010_B = []
    ruta_DJI_0018_A = []
    ruta_DJI_0036_A = []
    ruta_DJI_0043_A = []
    ruta_DJI_0051_A = []
    ruta_DJI_0075_A = []
    ruta_DJI_0083_A = []
    ruta_DJI_0098_A = []
    ruta_DJI_0101_A = []
    
    ruta_DJI_0010_A_mask_0 = []
    ruta_DJI_0010_B_mask_0 = []
    ruta_DJI_0018_A_mask_0 = []
    ruta_DJI_0036_A_mask_0 = []
    ruta_DJI_0043_A_mask_0 = []
    ruta_DJI_0051_A_mask_0 = []
    ruta_DJI_0075_A_mask_0 = []
    ruta_DJI_0083_A_mask_0 = []
    ruta_DJI_0098_A_mask_0 = []
    ruta_DJI_0101_A_mask_0 = []
    
    for i in range( len( lista_DJI_0010_A ) ):
        ruta_DJI_0010_A.append( os.path.join( DJI_0010_A_carpeta, lista_DJI_0010_A[i] ) )
        ruta_DJI_0010_B.append( os.path.join( DJI_0010_B_carpeta, lista_DJI_0010_B[i] ) )
        ruta_DJI_0018_A.append( os.path.join( DJI_0018_A_carpeta, lista_DJI_0018_A[i] ) )
        ruta_DJI_0036_A.append( os.path.join( DJI_0036_A_carpeta, lista_DJI_0036_A[i] ) )
        ruta_DJI_0043_A.append( os.path.join( DJI_0043_A_carpeta, lista_DJI_0043_A[i] ) )
        ruta_DJI_0051_A.append( os.path.join( DJI_0051_A_carpeta, lista_DJI_0051_A[i] ) )
        ruta_DJI_0075_A.append( os.path.join( DJI_0075_A_carpeta, lista_DJI_0075_A[i] ) )
        ruta_DJI_0083_A.append( os.path.join( DJI_0083_A_carpeta, lista_DJI_0083_A[i] ) )
        ruta_DJI_0098_A.append( os.path.join( DJI_0098_A_carpeta, lista_DJI_0098_A[i] ) )
        ruta_DJI_0101_A.append( os.path.join( DJI_0101_A_carpeta, lista_DJI_0101_A[i] ) )
    
    for i in range( len( lista_DJI_0010_A_mask_0 ) ):
        ruta_DJI_0010_A_mask_0.append( os.path.join( DJI_0010_A_mask_0_carpeta, lista_DJI_0010_A_mask_0[i] ) )
        ruta_DJI_0010_B_mask_0.append( os.path.join( DJI_0010_B_mask_0_carpeta, lista_DJI_0010_B_mask_0[i] ) )
        ruta_DJI_0018_A_mask_0.append( os.path.join( DJI_0018_A_mask_0_carpeta, lista_DJI_0018_A_mask_0[i] ) )
        ruta_DJI_0036_A_mask_0.append( os.path.join( DJI_0036_A_mask_0_carpeta, lista_DJI_0036_A_mask_0[i] ) )
        ruta_DJI_0043_A_mask_0.append( os.path.join( DJI_0043_A_mask_0_carpeta, lista_DJI_0043_A_mask_0[i] ) )
        ruta_DJI_0051_A_mask_0.append( os.path.join( DJI_0051_A_mask_0_carpeta, lista_DJI_0051_A_mask_0[i] ) )
        ruta_DJI_0075_A_mask_0.append( os.path.join( DJI_0075_A_mask_0_carpeta, lista_DJI_0075_A_mask_0[i] ) )
        ruta_DJI_0083_A_mask_0.append( os.path.join( DJI_0083_A_mask_0_carpeta, lista_DJI_0083_A_mask_0[i] ) )
        ruta_DJI_0098_A_mask_0.append( os.path.join( DJI_0098_A_mask_0_carpeta, lista_DJI_0098_A_mask_0[i] ) )
        ruta_DJI_0101_A_mask_0.append( os.path.join( DJI_0101_A_mask_0_carpeta, lista_DJI_0101_A_mask_0[i] ) )
    
    lista_training_set_RGB = (
        ruta_DJI_0010_B +
        ruta_DJI_0018_A +
        ruta_DJI_0036_A +
        ruta_DJI_0043_A +
        ruta_DJI_0075_A +
        ruta_DJI_0083_A +
        ruta_DJI_0098_A +
        ruta_DJI_0101_A
        )
    print '\nTraining images: '
    print ruta_DJI_0010_B[0]
    print ruta_DJI_0018_A[0]
    print ruta_DJI_0036_A[0]
    print ruta_DJI_0043_A[0]
    print ruta_DJI_0075_A[0]
    print ruta_DJI_0083_A[0]
    print ruta_DJI_0098_A[0]
    print ruta_DJI_0101_A[0]
    
    lista_test_set_RGB = ruta_DJI_0010_A + ruta_DJI_0051_A
    
    print '\nTesting images: '
    print ruta_DJI_0010_A[0] 
    print ruta_DJI_0051_A[0]
    
    lista_training_set_GT = (
        ruta_DJI_0010_B_mask_0 + 
        ruta_DJI_0018_A_mask_0 + 
        ruta_DJI_0036_A_mask_0 + 
        ruta_DJI_0043_A_mask_0 + 
        ruta_DJI_0075_A_mask_0 + 
        ruta_DJI_0083_A_mask_0 + 
        ruta_DJI_0098_A_mask_0 + 
        ruta_DJI_0101_A_mask_0
        )
    
    lista_test_set_GT = ruta_DJI_0010_A_mask_0 + ruta_DJI_0051_A_mask_0
    
    nombre_test_GT = lista_DJI_0010_A_mask_0 + lista_DJI_0051_A_mask_0
    
    return lista_training_set_RGB, lista_training_set_GT, lista_test_set_RGB, lista_test_set_GT, nombre_test_GT

def binarizar_predicciones_y_obtener_FP_FN_VP_VN(tensor_predict, test_images_GT):
    '''
    ------------------------------------
    | classified as:                   |
    |----------------------------------|
    | crop   | non-crop  | it really is|
    |--------|-----------|-------------|
    | TP     | FN        | crop        |
    |--------|-----------|-------------|
    | FP     | TN        | non-crop    |
    ------------------------------------

    True positive (TP):
    TP are the fig plant pixels correctly classified.

    True negative (TN):
    TN are non-plant pixels properly classified.

    False positive (FP):
    FP are pixels proposed as fig plant pixels but these do not really correspond to some part of the fig bushes
    
    False negative (FN):
    FN are fig plant pixels contained in the GT which are not detected by the system
    '''
    T = 0.5
    
    total_falsos_positivos = 0
    total_falsos_negativos = 0
    total_verdaderos_positivos = 0
    total_verdaderos_negativos = 0
    
    print '\n Binarizing the decoder output and computing the metrics for evaluation. . .\n'
    start_time = time.time()
    
    for num_img in range( tensor_predict.shape[0] ):        
        
        for i in range( tensor_predict.shape[1] ):
            
            for j in range( tensor_predict.shape[2] ):
                
                if ( tensor_predict[num_img, i, j, :] > T ):
                    tensor_predict[num_img, i, j, :] = 1
                else:
                    tensor_predict[num_img, i, j, :] = 0
                    
                if( tensor_predict[num_img, i, j, :] == 1 and test_images_GT[num_img, i, j, :] == 0):
                    total_falsos_positivos += 1
                    
                elif( tensor_predict[num_img, i, j, :] == 0 and test_images_GT[num_img, i, j, :] == 1):
                    total_falsos_negativos += 1
                    
                elif ( tensor_predict[num_img, i, j, :] == 1 and test_images_GT[num_img, i, j, :] == 1):
                    total_verdaderos_positivos += 1
                    
                elif( tensor_predict[num_img, i, j, :] == 0 and test_images_GT[num_img, i, j, :] == 0 ):
                    total_verdaderos_negativos += 1
    
    end_time = time.time()

    print '\n Process completed in: ', ( end_time - start_time ) / 60, ' min\n'
    
    return total_falsos_positivos, total_falsos_negativos, total_verdaderos_positivos, total_verdaderos_negativos

def exactitud( FP, FN, VP, VN ):
    return float( VP + VN ) / float( VP + VN + FP + FN )

def precision( FP, VP ):
    return float( VP ) / float( VP + FP )

def recuerdo( FN, VP ):
    return float( VP ) / float( VP + FN )

def especificidad( FP, VN ):
    return float( VN ) / float( VN + FP )

# valor predictivo negativo
def NPV( FN, VN ):
    return float( VN ) / float( VN + FN )

def medida_f( FP, FN, VP ):
    return float( 2*VP ) / float( 2*VP + FN + FP )

def ccm (VP, VN, FP, FN):
    return ( float ( VP * VN - FP * FN) ) / ( float( math.sqrt( ( VP + FP ) * ( VP + FN ) * ( VN + FP ) * ( VN + FN ) ) ) )

def TXTmetricas( VP, VN, FP, FN, exactitud, precision, recuerdo, especificidad, VPN, medida_f, matthews, total_set, total_training,  total_test, version, filas_training, cols_training, solapa, epocas):
    
    print '\n Saving the metrics in a txt file. . .\n'
    
    fp = open('Metricas_'+str(epocas)+'_epochs_'+version+'.txt','w')
    
    fp.write('Metrics report of the higNet-'+version+' on the *test set*\n')
    fp.write( '\nThe CNN was trained with a *dataset* of << '+str(total_set)+'>> divided in:\n\n' )
    fp.write('*Training set*: ' )
    fp.write( str(total_training) )
    fp.write('\n*Test set*: ')
    fp.write( str(total_test) )
    
    fp.write('\n\n Trained << '+str(epocas)+'epochs >>\n')
    fp.write('Using images of '+str(filas_training)+'x'+str(cols_training)+' with an overlapping of '+str(solapa)+'%\n')
    
    fp.write('\nTrue positives (pixels): ')
    fp.write( str(VP) )
    fp.write('\nTrue negatives (pixels): ')
    fp.write( str(VN) )
    fp.write('\nFalse positives (pixels): ')
    fp.write( str(FP) )
    fp.write('\nFalse negatives (pixels): ')
    fp.write( str(FN) )
    
    fp.write('\n\nAccuracy: ')
    fp.write( str(exactitud) )
    fp.write('\nPrecision: ')
    fp.write( str(precision) )
    fp.write('\nRecall: ')
    fp.write( str(recuerdo) )
    fp.write('\nSpecificity: ')
    fp.write( str(especificidad) )
    fp.write('\nNegative predictive value: ')
    fp.write( str(VPN) )
    fp.write('\nF-measure: ')
    fp.write( str(medida_f) )
    fp.write('\nMatthews Correlation Coefficient: ')
    fp.write( str(matthews) )
    
    fp.close()

def TXTAccuracy():
    
    print 'Writing two report files only with the accuracies'
    
    fp = open( 'Exactitudes_training_'+version+'.txt', 'a' )
    
    fp.write( str(exactitud_training) + '\n' )
    
    fp.close()
    
    fp = open( 'Exactitudes_test_'+version+'.txt', 'a' )
    
    fp.write( str(exactitud_test) + '\n' )
    
    fp.close()
    
def TXTLoss():
    
    print 'Writing two report files only with the losses'
    
    fp = open( 'Perdidas_training_'+version+'.txt', 'a' )
    
    fp.write( str(perdida_training) + '\n' )
    
    fp.close()
    
    fp = open( 'Perdidas_test_'+version+'.txt', 'a' )
    
    fp.write( str(perdida_test) + '\n' )
    
    fp.close()
    
def PLOTacc():

    x_epocas = range(1, epocas+1)
    
    plt.figure(figsize=(16, 9))
    plt.plot(x_epocas, lista_acc_train, 'b')
    plt.plot(x_epocas, lista_acc_test, 'g')
    # Here the zoom is removed
    # plt.axis([0,46,0,1])
    plt.ylabel(' A  c  c  u  r  a  c  y ')
    plt.xlabel(' E  p  o  c  h ')
    #plt.legend()
    #plt.grid(True)
    plt.savefig('higNet-'+version+'-ACCURACY.png', dpi = dpi_salida)

def PLOTloss():
    x_epocas = range(1, epocas+1)
    
    plt.figure(figsize=(16, 9))
    plt.plot(x_epocas, lista_loss_train, 'b')
    plt.plot(x_epocas, lista_loss_test, 'g')
    # Here the zoom is removed
    # plt.axis([0,46,0,1])
    plt.ylabel(' L  o  s  s ')
    plt.xlabel(' E  p  o  c  h ')
    #plt.legend()
    #plt.grid(True)
    plt.savefig('higNet-'+version+'-LOSS.png', dpi = dpi_salida)


total_set = 19380

total_training = 15504
total_validation = 1938
total_test = 1938 + total_validation
cortes_por_imagen = 1938

total_imgsotas = 10

filas_training = 128
cols_training = 128
solapa = 70

canales_training = 3
canales_label = 1

epocas = 120
dpi_salida = 200

version = 'v2-3'

original_set = '/home/juan/Documentos/higNet_dataset/higNet-v2-1_dataSet/higNet-v2-1_128x128_70Solapa_carpeta_por_imagen/'

autoencoder = load_model( 'higNet_v2-2_inicial.h5' )
autoencoder.summary()

lista_train_RGB, lista_train_GT, lista_test_RGB, lista_test_GT, nombres_test_GT = obtener_sets()

#Create the RGB dataset filled with zeros.
training_set_RGB = np.zeros((total_training, filas_training, cols_training, canales_training))
test_set_RGB = np.zeros((total_test, filas_training, cols_training, canales_training))

#Create the GT dataset filled with zeros.
training_set_GT = np.zeros((total_training, filas_training, cols_training))
test_set_GT = np.zeros((total_test, filas_training, cols_training))

# Read the training set and load it into the tensor

for i_imagen in range( len( lista_train_RGB ) ):
    
    training_set_RGB[ i_imagen ] = plt.imread( lista_train_RGB[ i_imagen ] )
    training_set_GT[ i_imagen ] = plt.imread( lista_train_GT[ i_imagen ] )

# Read the test set and load it into the tensor

for i_imagen in range( len( lista_test_RGB ) ):
    
    test_set_RGB[ i_imagen ] = plt.imread( lista_test_RGB[ i_imagen ] )
    test_set_GT[ i_imagen ] = plt.imread( lista_test_GT[i_imagen] )

print 'Reshaping the GTs'

training_set_GT = training_set_GT.astype('float32')
training_set_GT = np.reshape( training_set_GT, ( len( training_set_GT ), filas_training, cols_training, canales_label ) )

test_set_GT = test_set_GT.astype('float32')
test_set_GT = np.reshape( test_set_GT, ( len( test_set_GT ), filas_training, cols_training, canales_label ) )

lista_acc_train = []
lista_loss_train = []

lista_acc_test = []
lista_loss_test = []

for i_epoca in range(1, epocas+1):
    
    print 'Epoch # ', i_epoca, '/', epocas
    
    autoencoder.fit( training_set_RGB, training_set_GT, epochs = 1, shuffle=True )
    
    print 'Calculating the accuracy of the training set...'
    perdida_training, exactitud_training = autoencoder.evaluate( training_set_RGB, training_set_GT )
    
    print 'Calculating the accuracy of the test set...'
    perdida_test, exactitud_test = autoencoder.evaluate( test_set_RGB, test_set_GT )
    
    lista_acc_train.append( exactitud_training )
    lista_acc_test.append( exactitud_test )
    
    lista_loss_train.append( perdida_training )
    lista_loss_test.append( perdida_test )
    
    TXTAccuracy()
    TXTLoss()

PLOTacc()
PLOTloss()

print '\nSaving the input network. . .\n'
autoencoder.save('higNet-'+version+'-'+str(filas_training)+'x'+str(cols_training)+'-'+str(solapa)+'Solapa-'+str( epocas )+'epochs-'+ str(total_training) + 'ejemplos.h5')

print '\nPredicting the *test set*. . .\n'
predict_imgs = autoencoder.predict(test_set_RGB)
dir_predict = os.path.join( original_set,'predict'+str( epocas )+'epocas-'+  version )
os.mkdir( dir_predict )
print '\nSaving the predictions. . .\n'
for i_predict in range(total_test):
    
    imsave( os.path.join( dir_predict, nombres_test_GT[i_predict] ), predict_imgs[i_predict].reshape(filas_training, cols_training) )

total_falsos_positivos, total_falsos_negativos, total_verdaderos_positivos, total_verdaderos_negativos = binarizar_predicciones_y_obtener_FP_FN_VP_VN( predict_imgs, test_set_GT )

print '\nCalculating the metrics. . .\n'
acc = exactitud(total_falsos_positivos, total_falsos_negativos, total_verdaderos_positivos, total_verdaderos_negativos)
preci = precision( total_falsos_positivos, total_verdaderos_positivos )
recall = recuerdo( total_falsos_negativos, total_verdaderos_positivos )
especi = especificidad( total_falsos_positivos, total_verdaderos_negativos )
# negative predictive value
VPN = NPV( total_falsos_negativos, total_verdaderos_negativos )
f_medida = medida_f( total_falsos_positivos, total_falsos_negativos, total_verdaderos_positivos )
matthews = ccm( total_verdaderos_positivos, total_verdaderos_negativos, total_falsos_positivos, total_falsos_negativos )

TXTmetricas(total_verdaderos_positivos, total_verdaderos_negativos, total_falsos_positivos, total_falsos_negativos, acc, preci, recall, especi, VPN, f_medida, matthews, total_set, total_training, total_test, version, filas_training, cols_training, solapa, epocas)

dir_predict = os.path.join( original_set,'predict'+str( epocas )+'-'+ version +'-epocas_bin_dot5' )
os.mkdir( dir_predict )

print '\nSaving the binarized predictions. . .\n'
for i_predict in range(total_test):
    
    imsave( os.path.join( dir_predict, nombres_test_GT[i_predict] ), predict_imgs[i_predict].reshape(filas_training, cols_training) )
