import keras

# initializer =  keras.initializers.glorot_uniform(seed=0)
initializer = keras.initializers.glorot_normal()



"""
Creates Residual Network with 50 layers
"""
def create_model(input_shape=(64, 64, 3), classes=1):
    # Define the input as a tensor with shape input_shape
    X_input = keras.layers.Input(input_shape)

    # Zero-Padding
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', 
                            kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    # Stage 5
    X = convolutional_block(X, f = 3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = keras.layers.AveragePooling2D(pool_size=(2, 2))(X)
    
    # output layer
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(classes, activation='sigmoid', name='fc{}'
                            .format(classes), kernel_initializer=initializer)(X)
    
    # Create model
    model = keras.models.Model(inputs=X_input, outputs=X, name='resnet50')

    return model

"""
Identity Block of ResNet
"""
def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding='valid', 
                            name=conv_name_base + '2a', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.5)(X)
    
    # Second component of main path
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1,1), padding='same', 
                            name=conv_name_base + '2b', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.5)(X)

    # Third component of main path
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1,1), padding='valid', 
                            name=conv_name_base + '2c', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    
    return X

"""
Convolutional Block of ResNet
"""
def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = keras.layers.Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', 
                            padding='valid', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.5)(X)
    
    # Second component of main path
    X = keras.layers.Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', 
                            padding='same', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Dropout(0.5)(X)

    # Third component of main path
    X = keras.layers.Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', 
                            padding='valid', kernel_initializer=initializer)(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = keras.layers.Conv2D(F3, (1, 1), strides=(s,s), name=conv_name_base + '1', 
                                    padding='valid', kernel_initializer=initializer)(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    
    return X
   