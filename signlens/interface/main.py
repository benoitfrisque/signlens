from signlens.model.model import *
from signlens.params import *
from signlens.preprocessing.data import *
from signlens.preprocessing.preprocess import *
from sklearn.model_selection import train_test_split


def preprocess()-> None:
    train=load_data_subset_csv(balanced=True)
    y=label_dictionnary(train)
    X, X_test, y_t, y_test = train_test_split(train.file_path, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_t, test_size=0.2, random_state=42, stratify=y_t)
    X_train_sparse=group_pad_sequences(X_train)
    X_val_sparse=group_pad_sequences(X_val)
    X_test_sparse=group_pad_sequences(X_test)
    X_train_data=np.array([matrix.toarray().reshape(100, 75, 3) for matrix in X_train_sparse])
    X_val_data=np.array([matrix.toarray().reshape(100, 75, 3) for matrix in X_val_sparse])
    X_test_data=np.array([matrix.toarray().reshape(100, 75, 3) for matrix in X_test_sparse])
    return X_train_data,X_val_data,X_test_data,y_train,y_val,y_test

def train():
    X_train_data,X_val_data,X_test_data,y_train,y_val,y_test=preprocess()
    model=initialize_model(num_classes=y.shape[1])
    model=compile_model(model)
