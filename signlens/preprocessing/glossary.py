import pandas as pd
import os

from signlens.params import *

def write_glossary():

    if os.path.exists(GLOSSARY_CSV_PATH):
        response = input("Glossary already exists, are you sure you want to overwrite it? (y/n) ")
        if response.lower() == 'y':
            # Overwrite the glossary
            print("Overwriting the glossary...")
        else:
            print("Not overwriting the glossary.")
            return

    # Load glossary from landmarks files
    data_landmarks = pd.read_csv(TRAIN_CSV_PATH) # load the specified document
    gloss_landmarks = pd.DataFrame(data_landmarks['sign'].unique())\
            .rename(columns={0:'sign'})
    gloss_landmarks['sign_lower'] = gloss_landmarks['sign'].str.lower()
    gloss_landmarks.sort_values(by='sign', inplace=True, ignore_index=True)

    # Load glossary from videos
    data_videos = pd.read_json(WLASL_JSON_PATH)
    gloss_videos = pd.DataFrame(data_videos['gloss'].unique())\
            .rename(columns={0:'sign_videos'})
    gloss_videos['sign_videos_lower'] = gloss_videos['sign_videos'].str.lower()

    # Merge the two glossaries
    merged_df = gloss_videos.merge(gloss_landmarks, left_on='sign_videos_lower', right_on='sign_lower').reset_index()
    merged_signs = merged_df.sign

    # Take the missing signs from the landmarks glossary to append them
    missing_signs = gloss_landmarks[~gloss_landmarks.sign.isin(merged_signs)].sign
    all_signs = pd.concat([merged_signs , missing_signs]).reset_index(drop=True)

    all_signs.to_csv(GLOSSARY_CSV_PATH, index=True)
    print("✅ Glossary written to", GLOSSARY_CSV_PATH)

def load_glossary(csv_path=GLOSSARY_CSV_PATH):
    """
    Load a glossary from a CSV file into a pandas DataFrame.

    Parameters:
    - csv_path (str): The file path to the CSV file containing the glossary. Default is GLOSSARY_CSV_PATH.

    Returns:
    pandas.DataFrame: A DataFrame containing the loaded glossary data.
    """
    return pd.read_csv(csv_path, index_col=0)


def load_glossary_decoding(csv_path=GLOSSARY_DECODING_CSV_PATH):
    """
    Load a glossary from a CSV file into a pandas DataFrame. The values are slightly corrected compared to the original glossary.
    For example, hesheit is replaced by he / she / it. thankyou is replaced by thank you.

    Parameters:
    - csv_path (str): The file path to the CSV file containing the glossary. Default is GLOSSARY_DECODING_CSV_PATH.

    Returns:
    pandas.DataFrame: A DataFrame containing the loaded glossary data.
    """
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=0)
    else:
        print("Glossary decoding file not found. Loading the original glossary.")
        return pd.read_csv(GLOSSARY_CSV_PATH, index_col=0)



if __name__ == "__main__":
    write_glossary()
