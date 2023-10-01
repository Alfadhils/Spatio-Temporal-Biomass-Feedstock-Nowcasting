import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def create_sequences(data, seq_length, image=True):
    sequences = []
    targets = []
    for i in range(len(data)-seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    X = np.array(sequences)
    y = np.array(targets)
    if image :
        return X[..., np.newaxis], y
    return X, y

def save_model(model, name):
    model_json = model.to_json()
    with open('saved_models/'+str(name)+'.json', "w") as json_file:
        json_file.write(model_json)

    model.save_weights('saved_models/'+str(name)+'.h5')
    print('Save Successful!')

def real_eval(dfreal, predictions):
    mae_2018 = mean_absolute_error(dfreal['2018'], predictions['2018'])
    mae_2019 = mean_absolute_error(dfreal['2019'], predictions['2019'])
    return mae_2018,mae_2019

def get_dfbio_ts(dfbio):
    dfbio_t = dfbio[['2010','2011','2012','2013','2014','2015','2016','2017']].T
    dfbio_t.index = pd.to_datetime(dfbio_t.index)
    dfbio_t.columns = dfbio_t.columns.astype(str)
    return dfbio_t

def get_mask(dfbio, height, width, channels):
    mask = np.zeros((height, width, channels))
    for index, row in dfbio.iterrows():
        y, x = int(row['Latitude']), int(row['Longitude'])
        mask[y, x] = 1

    return mask

def plot_loss(history, offset=10, val=True):
    plt.figure(figsize=(10, 5))

    train_loss = history.history['loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs[offset:], train_loss[offset:], 'b', label='Training Loss')

    if val :
        val_loss = history.history['val_loss']
        plt.plot(epochs[offset:], val_loss[offset:], 'r', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def lr_scheduler_lstm(epoch, initial_lr=0.001, drop_factor=0.5, epochs_drop=[100, 200]):
    if epoch in epochs_drop:
        new_lr = initial_lr * drop_factor
        print(f'Learning rate changed to {new_lr} for epoch {epoch+1}')
        return new_lr
    else:
        return initial_lr
    
def lr_scheduler_conv(epoch, initial_lr=0.01, drop_factor=0.25, epochs_drop=[100, 200]):
    if epoch in epochs_drop:
        new_lr = initial_lr * drop_factor
        print(f'Learning rate changed to {new_lr} for epoch {epoch+1}')
        return new_lr
    else:
        return initial_lr

def lr_scheduler_convlstm(epoch, initial_lr=0.01, drop_factor=0.25, epochs_drop=[100, 200]):
    if epoch in epochs_drop:
        new_lr = initial_lr * drop_factor
        print(f'Learning rate changed to {new_lr} for epoch {epoch+1}')
        return new_lr
    else:
        return initial_lr