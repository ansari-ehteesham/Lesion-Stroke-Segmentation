import tensorflow as tf

class UNet2D:
    def __init__(self, 
                 nfilters = 32, 
                 nclassess = 1, 
                 final_class_activation = 'sigmoid',
                 activation = 'relu', 
                 kernel_initializer = 'he_normal', 
                 input_size = (128, 128, 3)):
        self.nfilter = nfilters
        self.nclassess = nclassess
        self.final_class_activation = final_class_activation
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.input_size = input_size


    
    def unet_model(self):
        inputs = tf.keras.layers.Input(self.input_size)

        # Encoder (Downsampling)
        conv1 = tf.keras.layers.Conv2D(self.nfilter*1, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=self.kernel_initializer)(inputs)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Conv2D(self.nfilter*1, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=self.kernel_initializer)(conv1)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(self.nfilter*2, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=self.kernel_initializer)(pool1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Conv2D(self.nfilter*2, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5), kernel_initializer=self.kernel_initializer)(conv2)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        drop2 = tf.keras.layers.SpatialDropout2D(0.3)(pool2)

        conv3 = tf.keras.layers.Conv2D(self.nfilter*4, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(drop2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Conv2D(self.nfilter*4, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv3)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(self.nfilter*8, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(pool3)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.Conv2D(self.nfilter*8, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv4)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        drop4 = tf.keras.layers.SpatialDropout2D(0.5)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottleneck
        conv5 = tf.keras.layers.Conv2D(self.nfilter*16, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(pool4)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)
        conv5 = tf.keras.layers.Conv2D(self.nfilter*16, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv5)
        conv5 = tf.keras.layers.BatchNormalization()(conv5)

        # Decoder (Upsampling)
        up6 = tf.keras.layers.Conv2DTranspose(self.nfilter*8, 2, strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv5)
        merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
        conv6 = tf.keras.layers.Conv2D(self.nfilter*8, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(merge6)
        conv6 = tf.keras.layers.BatchNormalization()(conv6)
        conv6 = tf.keras.layers.Conv2D(self.nfilter*8, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv6)
        conv6 = tf.keras.layers.BatchNormalization()(conv6)

        up7 = tf.keras.layers.Conv2DTranspose(self.nfilter*4, 2, strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv6)
        merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = tf.keras.layers.Conv2D(self.nfilter*4, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(merge7)
        conv7 = tf.keras.layers.BatchNormalization()(conv7)
        conv7 = tf.keras.layers.Conv2D(self.nfilter*4, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv7)
        conv7 = tf.keras.layers.BatchNormalization()(conv7)
        drop7 = tf.keras.layers.SpatialDropout2D(0.3)(conv7)

        up8 = tf.keras.layers.Conv2DTranspose(self.nfilter*2, 2, strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(drop7)
        merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = tf.keras.layers.Conv2D(self.nfilter*2, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(merge8)
        conv8 = tf.keras.layers.BatchNormalization()(conv8)
        conv8 = tf.keras.layers.Conv2D(self.nfilter*2, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv8)
        conv8 = tf.keras.layers.BatchNormalization()(conv8)

        up9 = tf.keras.layers.Conv2DTranspose(self.nfilter*1, 2, strides=(2, 2), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv8)
        merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = tf.keras.layers.Conv2D(self.nfilter*1, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(merge9)
        conv9 = tf.keras.layers.BatchNormalization()(conv9)
        conv9 = tf.keras.layers.Conv2D(self.nfilter*1, 3, activation=self.activation, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), kernel_initializer=self.kernel_initializer)(conv9)
        conv9 = tf.keras.layers.BatchNormalization()(conv9)
        drop9 = tf.keras.layers.SpatialDropout2D(0.5)(conv9) 

        # Output layer
        outputs = tf.keras.layers.Conv2D(self.nclassess, 1, activation=self.final_class_activation)(drop9)  

        model = tf.keras.models.Model(inputs, outputs)
        return model