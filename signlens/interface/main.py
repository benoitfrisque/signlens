from sklearn.model_selection import train_test_split
from utils.model_utils import *
from colorama import Fore, Style

from signlens.params import *
from signlens.preprocessing.data import *
from signlens.preprocessing.preprocess import *
from signlens.model.model import *


def preprocess():
    print(Fore.MAGENTA + "\n⭐️ Use case: preprocess" + Style.RESET_ALL)

    train = load_data_subset_csv(balanced=True)
    X_files = train.file_path
    y = encode_labels(train.sign)
    X_train_files, X_val_files, y_train, y_val = train_test_split(X_files, y, test_size=0.2, random_state=42, stratify=y)
    X_train = preprocess_and_pad_sequences_from_pq_list(X_train_files)
    X_val = preprocess_and_pad_sequences_from_pq_list(X_val_files)

    return X_train, X_val, y_train, y_val


def train(X_train, X_val, y_train, y_val, paths, patience=10, epochs=10, verbose=1, batch_size=32, validation_data=None, name_model=None):
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    name_model = input(
        Fore.RED + "Enter the name of the model you want to load (if you want to reset the model press enter):" + Style.RESET_ALL)
    model = load_model(name_model)
    if model is None:
        model = initialize_model(num_classes=NUM_CLASSES)
    model = compile_model(model)
    model, history = train_model(model, X_train, y_train,
                                 patience=patience,
                                 epochs=epochs,
                                 verbose=verbose,
                                 batch_size=batch_size,
                                 validation_data=[X_val, y_val],
                                 model_save_epoch_path=paths["model_each_epoch_path"]
                                 )

    val_accuracy = np.max(history.history['val_accuracy'])
    val_loss = np.max(history.history['val_loss'])

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X_train),
    )

    save_results(params=params,
                 metrics=dict(val_accuracy=val_accuracy, val_loss=val_loss),
                 params_path=paths["params_path"],
                 metrics_path=paths["metrics_path"]
                 )

    save_model(model=model, model_path=paths["model_path"])

    return paths


def evaluate(name_model=None):
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    name_model = input(
        Fore.RED+"Enter the name of the model you want to load:"+Style.RESET_ALL)

    model, path = load_model(name_model)
    assert model is not None

    test_data = load_data_subset_csv(balanced=True, csv_path=TRAIN_TEST_CSV_PATH)
    X_test = preprocess_and_pad_sequences_from_pq_list(test_data.file_path)
    y_test = encode_labels(test_data.sign)

    metrics_dict = evaluate_model(model=model, X=X_test, y=y_test)
    accuracy = metrics_dict['accuracy']
    loss = metrics_dict['loss']

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X_test),
    )

    save_results(params=params,
                 metrics=dict(val_accuracy=accuracy, val_loss=loss),
                 params_path=os.path.join(path, "params"),
                 metrics_path=os.path.join(path, "metrics"),
                 mode="evaluate")

    print("✅ evaluate() done \n")


def main():
    paths = create_folder_model()
    unique_train_test_split()
    X_train, X_val, y_train, y_val = preprocess()
    train(X_train, X_val, y_train, y_val, paths)
    evaluate()

if __name__ == "__main__":
    main()
