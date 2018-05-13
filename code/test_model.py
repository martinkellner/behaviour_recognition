from utils import *
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix

def test_dataset(path_to_model, path_to_csv):
    X_train, y_train = get_all_sequences_in_memory('train', path_to_csv=path_to_csv)
    X_test, y_test = get_all_sequences_in_memory('test', path_to_csv=path_to_csv)

    max_sq_len_test = max_sequence_len(X_train)
    max_sq_len_train = max_sequence_len(X_test)

    maxlen = max(max_sq_len_test, max_sq_len_train)

    X_test = pad_sequences(X_test, maxlen=maxlen, dtype='float32', value=0.0)

    model = load_model(path_to_model)
    result = model.evaluate(X_test, y_test)
    print(result)
    print(model.metrics_names)
    pred = model.predict(X_test)

    decode_pred = []
    decode_test = []
    for idx in range(len(pred)):

        decode_temp = [0.0 for _ in range(len(pred[0]))]
        decode_temp[np.argmax(pred[idx])] = 1.0

        decode_pred.append(hot_one(decode_temp))
        decode_test.append(hot_one(y_test[idx].tolist()))

    print(confusion_matrix(decode_test, decode_pred))


def predict(path_to_model, x):



path_to_csv='../data/features/pca_500/data.csv'
#path_to_model = '../data/checkpoints/lstm/sequences_pca_e500_b5.hdf5'
path_to_model = '../data/checkpoints/mlp/sequences_pca500_e12_b5.hdf5'
test_data(path_to_model, path_to_csv)





