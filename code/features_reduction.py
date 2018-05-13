import os
from utils import *
from sklearn import decomposition
from sklearn import discriminant_analysis
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences


def my_lda(path_to_csv, path_to_save, k=100):
    # Array to all features vector
    data_x, data_y = [], []
   # Remember file names of sequences and their number of features vertors
    memory = []
    # Read csv
    dataframe = pd.read_csv(path_to_csv)

    for idx, row in dataframe.iterrows():
        filename = row['0']
        sequences = get_sequence(filename)

        data_x.append(sequences.reshape(sequences.shape[0]*sequences.shape[1]))
        data_y.append(int(row['1']))

        memory.append(filename)

    maxlen = max_sequence_len(data_x)
    data_x = pad_sequences(data_x, maxlen=maxlen, dtype='float32', value=0.0)

    # Data standardization
    data_x = np.array(data_x)
    print('Data shape: ', data_x.shape)
    # LDA: dimensionality reduction
    lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', n_components=k)
    lda_data = lda.fit_transform(data_x, data_y)
    print(lda_data)
    # Save data back to sequences
    for idx in range(len(memory)):
        filename = path_to_save + memory[idx][24:]
        sequences = lda_data[idx]

        np.save(filename, sequences)

        # To be safe
        filename = None
        sequences = None

def my_pca(path_to_csv, path_to_save, k=100, save_pca=False):

    # Array to all features vector
    data = []
    # Remember file names of sequences and their number of features vertors
    memory = []
    # Read csv
    dataframe = pd.read_csv(path_to_csv)

    for idx, row in dataframe.iterrows():
        filename  = row['0']
        sequences = get_sequence(filename)
        cout_seq  = sequences.shape[0]

        for sequence in sequences:
            data.append(sequence)

        memory.append([filename, cout_seq])

    # Data standardization
    data = np.array(data)
    data_std = StandardScaler().fit_transform(data)

    # PCA: dimensionality reduction
    pca  = decomposition.PCA(n_components=k)
    pca_data = pca.fit_transform(data_std)

    # Save data back to sequences
    i = 0
    for item in memory:

        filename = path_to_save + item[0][17:]
        sequences = pca_data[i:i+item[1]]

        np.save(filename, sequences)

        i += item[1]

        # To be safe
        filename = None
        sequences = None

if __name__ == '__main__':
    #my_pca('../data/features/data.csv', '../data/features/pca_10/', k=10)
    my_lda('../data/features/pca_500/data.csv', '../data/features/lda_30/', k=20)
    create_data_csv('../data/features/lda_30/')
    split_data('../data/features/lda_30/data.csv')













