import keras
import numpy as np
import pandas as pd 
from sklearn import metrics


# load trained model
resnet50 = keras.models.load_model('models/resnet50.h5')

datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# data path
test_data_path= r'C:\Users\xx\Anaconda3\envs\yy\resnet\datasets\NPDI\test'
train_data_path= r'C:\Users\xx\Anaconda3\envs\yy\resnet\datasets\NPDI\train'


# image generator
train_generator = datagen.flow_from_directory(train_data_path,
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode= 'binary'  
                                                
                                                        )     

test_generator = datagen.flow_from_directory(test_data_path,
                                                        target_size=(64, 64),
                                                        batch_size=1,
                                                        class_mode= None , 
                                                        shuffle=False
                                                        )      

test1_generator = datagen.flow_from_directory(test_data_path,
                                                        target_size=(64, 64),
                                                        batch_size=1,
                                                        class_mode= 'binary' , 
                                                        shuffle=False
                                                        )                      


test_generator.reset()

# predict the test dataset
print('')
print('==predict the test set==')
print('')



Y_pred = resnet50.predict_generator(test_generator, steps=3000, verbose=1)

print('')
print('Y_pred')
print('')

print(Y_pred)
# save Y_pred  
df_Y_pred= pd.DataFrame(Y_pred)
df_Y_pred.to_excel('Y_Pred.xlsx')


print('')
print('n_pred') 
print('')

n_pred = np.where(Y_pred> 0.5, 1, 0)
print(n_pred)

print('')
print('reshape n_pred')   
print('')

y=np.reshape(n_pred,(1,3000))
print(y)

print('')
print('separate to 3000 array')
print('')

d = np.concatenate(n_pred,axis=0)
print(d)

print('')
print('==predictions table==')
print('')

y_pred = np.concatenate(n_pred,axis=0)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in y_pred]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_excel('Prediction.xlsx', index= True)
print(results)


# Performance parameter
print ('')
print ('==performance parameter==')
print ('')
clf_rep = metrics.precision_recall_fscore_support(test1_generator.classes, y_pred)
out_dict = {
                 "precision" :clf_rep[0].round(5)
                 ,"recall" : clf_rep[1].round(5)
                 ,"f1-score" : clf_rep[2].round(5)
                 ,"support" : clf_rep[3]
                 }
out_df = pd.DataFrame(out_dict, index = [ 'npe', 'porn'] )
avg_tot = (out_df.apply(lambda x: round(x.mean(), 5) if x.name!="support" else  round(x.sum(), 2)).to_frame().T)
avg_tot.index = ["avg/total"]
out_df = out_df.append(avg_tot)
print (out_df)
out_df.to_excel('Precision recall fscore support.xlsx', index= True)


# confusion matrix
print ('')
print ('==confusion matrix==')
print ('')
clf_cm = metrics.confusion_matrix(test1_generator.classes, y_pred)
out_dict_cm = {
                 "npe" :clf_cm[0]
                 ,"porn" : clf_cm[1]
                
                 
                 }

out_df_cm = pd.DataFrame(out_dict_cm, index = ['predicted as npe', 'predicted as porn']).transpose()
print (out_df_cm)
out_df_cm.to_excel('conf matrix.xlsx', index= True)


# accuracy
clf_acc = metrics.accuracy_score(test1_generator.classes, y_pred)
out_dict_acc = {
                 "" : [clf_acc]
                 }
out_df_acc = pd.DataFrame(out_dict_acc , index = ['acc'] )
print (out_df_acc)
out_df_acc.to_excel('acc.xlsx', index= True)







