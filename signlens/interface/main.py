from signlens.model.model import *
from signlens.params import *
from signlens.preprocessing.data import *
from signlens.preprocessing.preprocess import *
from sklearn.model_selection import train_test_split
from utils.model_utils import *

def preprocess():
    unique_train_test_split()
    train=load_data_subset_csv(balanced=True)
    y=label_dictionnary(train)
    X_train, X_val, y_train, y_val = train_test_split(train.file_path, y, test_size=0.2, random_state=42, stratify=y)
    X_train=group_pad_sequences(X_train)
    X_val=group_pad_sequences(X_val)
    return X_train,X_val,y_train,y_val

def train():
    X_train,X_val,y_train,y_val=preprocess()
    model = load_model()
    if model is None:
        model=initialize_model(num_classes=y_train.shape[1])
    model=compile_model(model)
    model, history=train_model(model,X_train,y_train,patience=10,epochs=10,verbose=1,batch_size=32,validation_data=[X_val,y_val])

    val_accuracy = np.max(history.history['val_accuracy'])
    val_loss = np.max(history.history['val_loss'])

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X_train),
    )
    save_results(params=params, metrics=dict(val_accuracy=val_accuracy,val_loss=val_loss))
    save_model(model=model)


def evaluate():

    test=load_data_subset_csv(balanced=True,csv_path=TRAIN_TEST_CSV_PATH)
    y_test=label_dictionnary(test)
    X_test=group_pad_sequences(test.file_path)
    model = load_model()
    assert model is not None
    metrics_dict = evaluate_model(model=model, X=X_test, y=y_test)
    accuracy = metrics_dict['accuracy']
    loss = metrics_dict['loss']

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X_test),
    )
    save_results(params=params, metrics=dict(val_accuracy=accuracy,val_loss=loss))
    print("✅ evaluate() done \n")
