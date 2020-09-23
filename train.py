import os
import click
import logging
import time
import keras
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
from model import create_model

K.set_image_data_format('channels_last')

"""
Train Model [optional args]
"""
@click.command(name='Training Configuration')
@click.option(
    '-lr', 
    '--learning-rate', 
    default=0.0005, 
    help='Learning rate for minimizing loss during training'
)
@click.option(
    '-bz',
    '--batch-size',
    default=64,
    help='Batch size of minibatches to use during training'
)
@click.option(
    '-ne', 
    '--num-epochs', 
    default=20, 
    help='Number of epochs for training model'
)
@click.option(
    '-se',
    '--save-every',
    default=1,
    help='Epoch interval to save model checkpoints during training'
)
@click.option(
    '-tb',
    '--tensorboard-vis',
    is_flag=True,
    help='Flag for TensorBoard Visualization'
)
@click.option(
    '-ps',
    '--print-summary',
    is_flag=True,
    help='Flag for printing summary of the model'
)


def train(learning_rate, batch_size, num_epochs, save_every, tensorboard_vis, print_summary):
    setup_paths()

    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    get_gen = lambda x: datagen.flow_from_directory(
        'datasets/NPDI/{}'.format(x),
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary'    
    )

    #generator objects
    train_generator = get_gen('train')
    val_generator = get_gen('val')
    test_generator = get_gen('test')
    
    if os.path.exists('models/resnet50.h5'):
        # load model
        logging.info('loading pre-trained model')
        resnet50 = keras.models.load_model('models/resnet50.h5')
    else:
        # create model
        logging.info('creating model')
        resnet50 = create_model(input_shape=(64, 64, 3), classes=1)
    
    # freeze certain layer    
#    resnet50.trainable = False
#    resnet50.layers[-41].trainable = True
    
    optimizer = keras.optimizers.Adam(learning_rate)
    resnet50.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    if print_summary:
        resnet50.summary()
               
    callbacks = configure_callbacks(save_every, tensorboard_vis)
 
    # count certain amount of layer
#    print(len(resnet50.layers) -resnet50.layers.index(resnet50.get_layer("res5a_branch2a")))
    
    #save training time per-epoch
    class TimeHistory(keras.callbacks.Callback):
        
        def __init__(self, logs={}):
           self.times = []
        
        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start =time.time()

        def on_epoch_end(self, epoch, logs={}):
            self.times.append(time.time() - self.epoch_time_start)  
          
    time_callback = TimeHistory()
    
    callbacks.extend([time_callback])
    
    # train model
    logging.info('training model')
    
     # display summary of model
#    logging.info('model summary')
#    resnet50.summary()
    
    archi=resnet50.fit_generator(
        train_generator,
        steps_per_epoch=  12000//batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=val_generator,
        validation_steps= 1200//batch_size,
        shuffle=True,
        callbacks=callbacks
    )
  
   
    # save model
    logging.info('Saving trained model to `models/resnet50.h5`')
    resnet50.save('models/resnet50.h5')

    #evaluate model
    logging.info('evaluating model')
    preds = resnet50.evaluate_generator(
        test_generator,
        steps= 3000//batch_size,
        verbose=1
    )
    logging.info('test loss: {:.4f} - test acc: {:.4f}'.format(preds[0], preds[1]))

    keras.utils.plot_model(resnet50, to_file='models/resnet50.png')

    
    
    # visualizing the training and validation accuracy
    logging.info('training-validation acc graph') 
    # training-validation acc
    plt.plot(archi.history['acc'])
    plt.plot(archi.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # save image
    plt.savefig('train vs val acc.png')
    # show image
    plt.show()
      
    # visualizing the training and validation loss
    logging.info('training-validation loss graph') 
    # training-validation loss
    plt.plot(archi.history['loss'])
    plt.plot(archi.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='upper left')
    # save image
    plt.savefig('train vs val loss.png')
    # show image
    plt.show()
    
    # Displaying training time per-epoch
    logging.info('training time per-epoch') 
    arr=time_callback.times
    time_table= {'Time of x-th epoch': arr }
    df = pd.DataFrame(data=time_table)
    print(df)
    # save to excel
    df.to_excel('Time of x-th epoch.xlsx', index= True)
    
    # Displaying total training time 
    _sum=sum(arr)
    out_dict_sum_time = {
                 "" : [_sum]
                 }
    out_df_sum_time = pd.DataFrame(out_dict_sum_time , index = ['Total of computational time'] )
    print (out_df_sum_time)
    # save to excel
    out_df_sum_time.to_excel('Total of computational time.xlsx', index= True)
    
    # save training history
    logging.info('training history')
    hist_df = pd.DataFrame(archi.history)
    print (hist_df)
    # save to excel
    hist_df.to_excel('history.xlsx', index= True)
    # or save to csv: 
#    hist_csv_file = 'history.xlsx'
#    with open(hist_csv_file, mode='w') as f:
#        hist_df.to_excel(f)
    

"""
Configure Callbacks for Training
"""
def configure_callbacks(save_every=1, tensorboard_vis=False):
    # checkpoint models only when `val_loss` impoves
    saver = keras.callbacks.ModelCheckpoint(
        'models/ckpts/model.ckpt',
        monitor='val_loss',
        save_best_only=True,
        period=save_every,
        verbose=1
    )
        
#def configure_callbacks( tensorboard_vis=False):
#    # checkpoint models only when `val_loss` impoves
#    saver = keras.callbacks.ModelCheckpoint(
#        'models/ckpts/model.ckpt',
#        monitor='val_loss',
#        save_best_only=True,
#        mode= 'min',
#        verbose=1
#    )   
    
    # reduce LR when `val_loss` plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        verbose=1,
        min_lr=1e-10
    )

    # early stopping when `val_loss` stops improving
    early_stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=10, 
        verbose=1
    )
    
    callbacks = [saver]

    if tensorboard_vis:
        # tensorboard visualization callback
        tensorboard_cb = keras.callbacks.TensorBoard(
            log_dir='./logs',
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_cb)
    
    return callbacks

def setup_paths():
    if not os.path.isdir('models/ckpts'):
        if not os.path.isdir('models'):
            os.mkdir('models')
        os.mkdir('models/ckpts')

def main():
    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(
        format=LOG_FORMAT, 
        level='INFO'
    )

    try:
        train()
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()
