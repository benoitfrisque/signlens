from signlens.model.model import *
from signlens.params import *
from signlens.preprocessing.data import *
from signlens.preprocessing.preprocess import *
from sklearn.model_selection import train_test_split
from utils.model_utils import *

def preprocess():
    train=load_data_subset_csv(balanced=True)
    y=label_dictionnary(train)
    X, X_test, y_t, y_test = train_test_split(train.file_path, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_t, test_size=0.2, random_state=42, stratify=y_t)
    X_train=group_pad_sequences(X_train)
    X_val=group_pad_sequences(X_val)
    X_test=group_pad_sequences(X_test)
    return X_train,X_val,X_test,y_train,y_val,y_test

def train():
    X_train,X_val,X_test,y_train,y_val,y_test=preprocess()
    model = load_model()
    if model is None:
        model=initialize_model(num_classes=y_train.shape[1])
    model=compile_model(model)
    model, history=train_model(model,X_train,y_train,patience=10,verbose=1,batch_size=32,validation_data=[X_val,y_val])

    val_accuracy = np.max(history.history['val_accuracy'])
    val_loss = np.max(history.history['val_loss'])

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X_train),
    )
    save_results(params=params, metrics=dict(val_accuracy=val_accuracy,val_loss=val_loss))
    save_model(model=model)
    return X_test,y_test

def evaluate():
    X,y = train()
    model = load_model()
    print(X)
    print(y)
    assert model is not None
    metrics_dict = evaluate_model(model=model, X=X, y=y)
    accuracy = metrics_dict['accuracy']
    loss = metrics_dict['loss']

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X),
    )
    save_results(params=params, metrics=dict(val_accuracy=accuracy,val_loss=loss))
    print("✅ evaluate() done \n")
