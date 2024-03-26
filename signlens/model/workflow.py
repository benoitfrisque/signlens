import random
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from torch import rand

from signlens.params import *
from signlens.preprocessing.data import load_data_subset_csv, unique_train_test_split
from signlens.preprocessing.preprocess import pad_and_preprocess_sequences_from_pq_file_path_df, encode_labels
from signlens.model.model_architecture import initialize_model, compile_model, train_model, evaluate_model
from signlens.model.model_utils import save_results, save_model, load_model, create_model_folder


def preprocess(random_state=None):
    print(Fore.MAGENTA + Style.BRIGHT + "\n⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Data loading
    train = load_data_subset_csv(balanced=True, random_state=random_state)

    # Train test split
    X_files = train.file_path
    y = encode_labels(train.sign)
    X_train_files, X_val_files, y_train, y_val = train_test_split(X_files, y, test_size=0.2, stratify=y, random_state=random_state)

    # Preprocessing
    print(Fore.BLUE + f"\nPreprocessing {len(X_train_files)} training files..." + Style.RESET_ALL)
    X_train = pad_and_preprocess_sequences_from_pq_file_path_df(X_train_files)
    print(Fore.BLUE + f"\nPreprocessing {len(X_val_files)} validation files..." + Style.RESET_ALL)
    X_val = pad_and_preprocess_sequences_from_pq_file_path_df(X_val_files)

    return X_train, X_val, y_train, y_val


def train(X_train, y_train,epochs=EPOCHS, patience=20, verbose=1, batch_size=32, validation_data=None, shuffle=True):

    print(Fore.MAGENTA + Style.BRIGHT + "\n⭐️ Use case: train" + Style.RESET_ALL)

    new_model_required = ''
    while new_model_required.lower() not in ['y', 'n']:
        new_model_required = input("Do you want to train a new model from scratch? (y/n): ")

    if new_model_required.strip().lower() == 'y':
        paths = create_model_folder()
        model = initialize_model(num_classes=NUM_CLASSES)
    else:
        model_base_dir_pattern = input("Enter the name (or a part of the name) of the model you want to load: ").strip()
        model, paths = load_model(mode='most_recent', model_base_dir_pattern=model_base_dir_pattern, return_paths=True)

    model = compile_model(model)
    model, history = train_model(model, X_train, y_train,
                                 patience=patience,
                                 epochs=epochs,
                                 verbose=verbose,
                                 batch_size=batch_size,
                                 validation_data=validation_data,
                                 model_save_epoch_path=paths['iter'],
                                 shuffle=shuffle
                                 )

    val_accuracy = np.max(history.history['val_accuracy'])

    params = dict(
        context="train",
        training_frac=DATA_FRAC,
        row_count=len(X_train),
        num_classes=NUM_CLASSES,
    )

    save_results(params=params,
                 metrics=dict(val_accuracy=val_accuracy),
                 params_path=paths['params'],
                 metrics_path=paths['metrics'],
                 mode='train'
                 )

    save_model(model=model, model_path=paths['model'])

    return model, paths


def evaluate(random_state=None, model=None, paths=None):
    print(Fore.MAGENTA + Style.BRIGHT + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    if model is None:
        model_base_dir_pattern = input("Enter the name (or a part of the name) of the model you want to load: ").strip()
        model, paths = load_model(mode='most_recent', model_base_dir_pattern=model_base_dir_pattern, return_paths=True)
        assert model is not None

    test_data = load_data_subset_csv(balanced=True, csv_path=TRAIN_TEST_CSV_PATH, random_state=random_state)
    X_test_files = test_data.file_path

    # Preprocessing
    print(Fore.BLUE + f"\nPreprocessing {len(X_test_files)} testing files..." + Style.RESET_ALL)
    X_test = pad_and_preprocess_sequences_from_pq_file_path_df(X_test_files)
    y_test = encode_labels(test_data.sign)

    metrics_dict = evaluate_model(model, X_test, y_test)
    accuracy = metrics_dict['accuracy']

    params = dict(
        context="evaluate",
        row_count=len(X_test),
        num_classes=NUM_CLASSES,
    )

    save_results(params=params,
                 metrics=dict(val_accuracy=accuracy),
                 params_path=paths['params'],
                 metrics_path=paths['metrics'],
                 mode='evaluate')

    print(f'✅ evaluate() done \n')


def main(random_state=None):
    unique_train_test_split()
    X_train, X_val, y_train, y_val = preprocess(random_state=random_state)
    shuffle = (random_state is None) # shuffle in fit if random_state is None
    model, paths = train(X_train, y_train, validation_data=(X_val, y_val), shuffle=shuffle)
    evaluate(random_state=random_state, model=model, paths=paths)

if __name__ == "__main__":
    main()
