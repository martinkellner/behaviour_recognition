from keras.layers import Dense, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque
from utils import *
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.sequence import pad_sequences
from sklearn import svm
import os, time
from sklearn.metrics import accuracy_score

def lstm(nb_classes, input_shape):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False,
                   input_shape=input_shape,
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def lstm_pca_100(nb_classes, input_shape):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    model = Sequential()
    model.add(LSTM(100, return_sequences=False,
                   input_shape=input_shape,
                   dropout=0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def lstm_pca_500(nb_classes, input_shape):
    """Build a simple LSTM network. We pass the extracted features from
    our CNN to this model predomenently."""
    model = Sequential()
    model.add(LSTM(500, return_sequences=False,
                   input_shape=input_shape,
                   dropout=0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def mlp_pca_500(nb_classes, input_shape):
    # Build simple MLP. We pass the extracted features from our CNN to this model.
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(500))
    model.add(Dropout(0.5))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def mlp_pca_100(nb_classes, input_shape):
    # Build simple MLP. We pass the extracted features from our CNN to this model.
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def mlp_pca_10(nb_classes, input_shape):
    # Build simple MLP. We pass the extracted features from our CNN to this model.
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def mlp_lda(nb_classes, input_shape):
    # Build simple MLP. We pass the extracted features from our CNN to this model.
    model = Sequential()
    model.add(Dense(10, input_shape=input_shape))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def mlp_lda_2(nb_classes, input_shape):
    # Build simple MLP. We pass the extracted features from our CNN to this model.
    model = Sequential()
    model.add(Dense(10, input_shape=input_shape))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def train(model, train_data, test_data, callbacks, batch_size=32, epochs=100, verbose=1):

    X_train, y_train = train_data[0], train_data[1]
    X_test, y_test = test_data[0], test_data[1]

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=verbose,
        callbacks=callbacks,
        epochs=epochs,
    )

def svm_train():
    path_to_csv = '../data/features/lda_30/data.csv'
    train_x, train_y = get_all_svm(path_to_csv, 'train')

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(train_x, train_y)

    test_x, test_y = get_all_svm(path_to_csv, 'test')
    pred_y = clf.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print(acc)

svm_train()

'''X_train, y_train = get_all_sequences_in_memory('train', path_to_csv=path_to_csv)
X_test, y_test = get_all_sequences_in_memory('test', path_to_csv=path_to_csv)

#max_sq_len_test  = max_sequence_len(X_train)
#max_sq_len_train = max_sequence_len(X_test)

#maxlen = max(max_sq_len_test, max_sq_len_train)

#X_test  = pad_sequences(X_test, maxlen=maxlen, dtype='float32', value=0.0)
#X_train = pad_sequences(X_train, maxlen=maxlen, dtype='float32', value=0.0)

input_shape = X_train[0].shape

nb_classes = 6
metrics = ['accuracy']

print("Loading model.")
#model = lstm(nb_classes, input_shape)
#model = lstm_pca_100(nb_classes, input_shape)
#model = lstm_pca_500(nb_classes, input_shape)
#model = mlp_pca_500(nb_classes, input_shape)
#model = mlp_pca_10(nb_classes, input_shape)
#model = mlp_pca_100(nb_classes, input_shape)
#model = mlp_lda(nb_classes, input_shape)
model = mlp_lda_2(nb_classes, input_shape)


# Now compile the network.
optimizer = Adam(lr=1e-5, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                   metrics=metrics)

tb = TensorBoard(log_dir=os.path.join('data', 'logs', 'lstm'))

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=5)

timestamp = time.time()
csv_logger = CSVLogger('../data/logs/lstm' + '-' + 'lda_2_training_e200_b5-' + str(timestamp) + '.log')

checkpointer = ModelCheckpoint(
    filepath=('../data/checkpoints/lstm/sequences_lda_2_e200_b5.hdf5'),
    verbose=1,
    save_best_only=True)

train(model, [X_train, y_train], [X_test, y_test], [tb, early_stopper, csv_logger, checkpointer], epochs=500, batch_size=5)
'''