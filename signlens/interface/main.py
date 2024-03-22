from sklearn.model_selection import train_test_split
from utils.model_utils import *
from colorama import Fore, Style

from signlens.params import *
from signlens.preprocessing.data import *
from signlens.preprocessing.preprocess import  *
from signlens.model.model import *

def preprocess():
    unique_train_test_split()
    train = load_data_subset_csv(balanced=True)
    y = encode_labels(train.sign)
    X_train, X_val, y_train, y_val = train_test_split(
        train.file_path, y, test_size=0.2, random_state=42, stratify=y)
    X_train = group_pad_sequences(X_train)
    X_val = group_pad_sequences(X_val)
    return X_train, X_val, y_train, y_val


def train(name_model=None):

    paths = create_folder_model()
    X_train, X_val, y_train, y_val = preprocess()
    name_model = input(
        Fore.RED+"Enter the name of the model you want to load (if you want to reset the model press enter):"+Style.RESET_ALL)
    model = load_model(name_model)
    if model is None:
        import ipdb; ipdb.set_trace()
        model = initialize_model(num_classes=NUM_CLASSES)
    model = compile_model(model)
    model, history = train_model(model, X_train, y_train, patience=10, epochs=10, verbose=1, batch_size=32, validation_data=[
                                 X_val, y_val], model_save_epoch_path=paths["model_each_epoch_path"])
    val_accuracy = np.max(history.history['val_accuracy'])
    val_loss = np.max(history.history['val_loss'])

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X_train),
    )
    save_results(params=params, metrics=dict(val_accuracy=val_accuracy, val_loss=val_loss),
                 params_path=paths["params_path"], metrics_path=paths["metrics_path"])
    save_model(model=model, model_path=paths["model_path"])
    return paths


def evaluate(name_model=None):

    name_model = input(
        Fore.RED+"Enter the name of the model you want to load:"+Style.RESET_ALL)
    test = load_data_subset_csv(balanced=True, csv_path=TRAIN_TEST_CSV_PATH)
    y_test = encode_labels(test.sign)
    X_test = group_pad_sequences(test.file_path)
    model, path = load_model(name_model)
    assert model is not None
    metrics_dict = evaluate_model(model=model, X=X_test, y=y_test)
    accuracy = metrics_dict['accuracy']
    loss = metrics_dict['loss']

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X_test),
    )
    save_results(params=params, metrics=dict(val_accuracy=accuracy, val_loss=loss), params_path=os.path.join(
        path, "params"), metrics_path=os.path.join(path, "metrics"), mode="evaluate")
    print("✅ evaluate() done \n")
