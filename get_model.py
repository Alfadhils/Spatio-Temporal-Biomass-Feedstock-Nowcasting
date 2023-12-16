import numpy as np
import matplotlib.pyplot as plt
import utils

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, ConvLSTM2D, Conv2D, Conv2DTranspose, Conv3D, BatchNormalization, Flatten, Dense, Dropout, Lambda, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation, SeparableConv2D, MaxPooling2D, UpSampling2D, LSTM, GRU, add, LeakyReLU, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, Adadelta


def encoder(inputs, filters, pool_size):
    conv_pool = Conv2D(filters, (3, 3), padding='same')(inputs)
    conv_pool = BatchNormalization()(conv_pool)
    conv_pool = Activation('relu')(conv_pool)

    conv_pool = Conv2D(filters, (3, 3), padding='same')(conv_pool)
    conv_pool = BatchNormalization()(conv_pool)
    conv_pool = Activation('relu')(conv_pool)
    conv_pool = MaxPooling2D(pool_size=pool_size)(conv_pool)
    return conv_pool

def decoder(inputs, concat_input, filters, transpose_size):
    up = Concatenate()([Conv2DTranspose(filters, transpose_size, strides=(2, 2), padding='same')(inputs), concat_input])
    up = Conv2D(filters, (3, 3), padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)

    up = Conv2D(filters, (3, 3), padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    return up

class Unet:
    def __init__(self, dfbio, images_list):
        self.dfbio = dfbio
        self.images = images_list
        self.height, self.width = self.images.shape[1:]
        self.channels = 1

        lr_scheduler_callback = LearningRateScheduler(utils.lr_scheduler_conv)
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)
        self.callbacks = [lr_scheduler_callback]

    def prepare_data(self, seq_length=1, fit=False):
        self.seq_len = seq_length
        if not fit:
            train = self.images[:-1]
            val = self.images[-1:]

            X_train, y_train = utils.create_sequences(train, self.seq_len)
            X_val, y_val = utils.create_sequences(np.concatenate([train[-self.seq_len:], val]), self.seq_len)

            return X_train[:,0],y_train,X_val[:,0],y_val

        else :
            X_train, y_train = utils.create_sequences(self.images, self.seq_len)
            
            return X_train[:,0],y_train

    def get_model(self):
        inputs = Input(shape=(self.height, self.width, self.channels))

        x = Conv2D(16, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        e1 = encoder(x, 32, (2,2))
        e2 = encoder(e1, 64, (2,2))

        bridge = Conv2D(128, (3, 3), padding='same')(e2)
        bridge = BatchNormalization()(bridge)
        bridge = Activation('relu')(bridge)

        u1 = decoder(bridge, e1, 64, (2,2))
        u2 = decoder(u1, x, 32, (2,2))

        x = Conv2D(1, (1, 1), padding='same', activation='relu')(u2)

        multiply_layer = Lambda(lambda tensors: tensors[0] * tensors[1], output_shape=(self.height, self.width, self.channels))
        mask = utils.get_mask(self.dfbio, self.height, self.width, self.channels)
        outputs = multiply_layer([x, mask]) 

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mae')

        return model

    def eval(self, X_train, y_train, X_val, y_val):
        self.model = self.get_model()

        self.history = self.model.fit(X_train, y_train, epochs=250, batch_size=3, verbose=1, validation_data=(X_val, y_val), callbacks=[self.callbacks])

        self.val_loss = self.model.evaluate(X_val,y_val)

        print(f'Validation loss : {self.val_loss}')

        return self.history
        
    def fit(self, X_train, y_train):
        self.model = self.get_model()

        self.history = self.model.fit(X_train, y_train, epochs=250, batch_size=4, verbose=1, callbacks=[self.callbacks])

        self.train_loss = self.model.evaluate(X_train,y_train)

        print(f'Train loss : {self.train_loss}')

        return self.history
    
    def predict(self, selected_pix):
        predictions = []
        last = self.images[-1]
        window = last.reshape(1, last.shape[0], last.shape[1], 1)

        for i in range(2):
            prediction = self.model.predict(window)
            predictions.append(prediction)
            window = prediction.reshape(1, last.shape[0], last.shape[1], 1)

        preds = {'2018':[predictions[0][0,y,x,0] for x,y in selected_pix], '2019':[predictions[1][0,y,x,0] for x,y in selected_pix]}
        
        return predictions, preds
    

class LSTM_3 :
    def __init__(self, dfbio):
        self.dfbio = dfbio
        self.dfbio_t = utils.get_dfbio_ts(dfbio)

        lr_scheduler_callback = LearningRateScheduler(utils.lr_scheduler_lstm)
        early_stopping = EarlyStopping(monitor='val_loss', patience=150, verbose=1, restore_best_weights=True)
        self.callbacks = [lr_scheduler_callback]


    def prepare_data(self, seq_length=3, fit=False):
        self.seq_len = seq_length
        if not fit:
            train = self.dfbio_t[:-1].values
            val = self.dfbio_t[-1:].values

            X_train, y_train = utils.create_sequences(train, self.seq_len, image=False)
            X_val, y_val = utils.create_sequences(np.concatenate([train[-self.seq_len:], val]), self.seq_len, image=False)

            return X_train, y_train, X_val, y_val
        
        else :
            X_train, y_train = utils.create_sequences(self.dfbio_t.values, self.seq_len, image=False)

            return X_train, y_train

    def get_model(self):
        inputs = Input(shape=(self.seq_len, self.dfbio_t.shape[-1]))

        x = LSTM(units=256, activation='relu', return_sequences = True)(inputs)
        x = LSTM(units=256, activation='relu', return_sequences = True)(x)
        x = LSTM(units=128, activation='relu', return_sequences = False)(x)

        outputs = Dense(units=2418, activation='linear')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

        return model

    def eval(self, X_train, y_train, X_val, y_val):
        self.model = self.get_model()

        self.history = self.model.fit(X_train, y_train, epochs=300, batch_size=6, verbose=1, validation_data=(X_val, y_val), callbacks=[self.callbacks])

        val_loss = self.model.evaluate(X_val,y_val)

        print(f'Validation loss : {val_loss}')

        return self.history

    def fit(self, X_train, y_train):
        self.model = self.get_model()

        self.history = self.model.fit(X_train, y_train, epochs=300, batch_size=7, verbose=1, callbacks=[self.callbacks])

        train_loss = self.model.evaluate(X_train,y_train)

        print(f'Train loss : {train_loss}')

        return self.history

    def predict(self):
        predictions = []
        window = utils.get_dfbio_ts(self.dfbio)[-self.seq_len:].values
        window = window.reshape(1, window.shape[0], window.shape[1])

        for i in range(2):
            prediction = self.model.predict(window)
            predictions.append(prediction)
            window = np.vstack([window[0][1:], prediction]).reshape(1, window.shape[1], window.shape[2])


        preds = {'2018':predictions[0][0], '2019':predictions[1][0]}

        return predictions, preds

class ConvLSTM_3:
    def __init__(self, dfbio, images_list):
        self.dfbio = dfbio
        self.images = images_list
        self.height, self.width = self.images.shape[1:]
        self.channels = 1

        lr_scheduler_callback = LearningRateScheduler(utils.lr_scheduler_convlstm)
        self.callbacks = [lr_scheduler_callback]
    
    def prepare_data(self, seq_length=3, fit=False):
        self.seq_len = seq_length
        if not fit:
            train = self.images[:-1]
            val = self.images[-1:]

            X_train, y_train = utils.create_sequences(train, self.seq_len)
            X_val, y_val = utils.create_sequences(np.concatenate([train[-self.seq_len:], val]), self.seq_len)

            return X_train,y_train,X_val,y_val

        else :
            X_train, y_train = utils.create_sequences(self.images, self.seq_len)
            
            return X_train,y_train

    def get_model(self):
        inputs = Input(shape=(self.seq_len, self.height, self.width, self.channels))

        x = ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ConvLSTM2D(filters=64, kernel_size=(1, 1), padding='same', return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=self.channels, kernel_size=(1,1), activation='relu', padding='valid')(x)

        multiply_layer = Lambda(lambda tensors: tensors[0] * tensors[1], output_shape=(self.height, self.width, self.channels))
        mask = utils.get_mask(self.dfbio, self.height, self.width, self.channels)
        outputs = multiply_layer([x, mask]) 

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mae')

        return model

    def eval(self, X_train, y_train, X_val, y_val):
        self.model = self.get_model()

        self.history = self.model.fit(X_train, y_train, epochs=300, batch_size=1, verbose=1, validation_data=(X_val, y_val), callbacks=[self.callbacks])

        val_loss = self.model.evaluate(X_val,y_val)

        print(f'Validation loss : {val_loss}')

        return self.history
    
    def fit(self, X_train, y_train):
        self.model = self.get_model()

        self.history = self.model.fit(X_train, y_train, epochs=300, batch_size=1, verbose=1, callbacks=[self.callbacks])

        train_loss = self.model.evaluate(X_train,y_train)

        print(f'Train loss : {train_loss}')

        return self.history

    def predict(self, selected_pix):
        predictions = []

        last = self.images[-self.seq_len:]
        window = last.reshape(1, self.seq_len, last.shape[1], last.shape[2], 1)

        for i in range(2):
            prediction = self.model.predict(window)
            prediction[prediction < 0] = 0
            predictions.append(prediction)
            window = np.vstack([window[0][1:], prediction]).reshape(1, self.seq_len, last.shape[1], last.shape[2], 1)

        preds = {'2018':[predictions[0][0,y,x,0] for x,y in selected_pix], '2019':[predictions[1][0,y,x,0] for x,y in selected_pix]}
        return predictions, preds