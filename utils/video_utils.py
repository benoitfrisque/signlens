import os
import json
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2

#import pyarrow.csv as pv
#import pyarrow.parquet as pq

################################################################################
# Extract Landmarks to file and dataframe
################################################################################
# TO DO: Dataframe format is packed in columns, need to expand it
def serialize_landmarks(landmark_list):
    '''
    Serialize a list of landmarks into a dictionary format.

    Args:
        landmark_list (list): A list of landmarks.

    Returns:
        list: A list of dictionaries, where each dictionary represents a landmark and contains the following keys:
            - 'x': The x-coordinate of the landmark.
            - 'y': The y-coordinate of the landmark.
            - 'z': The z-coordinate of the landmark.
            - 'visibility': The visibility of the landmark (if available), otherwise np.nan.

    '''
    if landmark_list is None:
        return [{'x': np.nan, 'y': np.nan, 'z': np.nan}]
    landmarks = []
    for landmark in landmark_list.landmark:
        landmarks.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        })
    return landmarks

def process_video_to_landmarks(video_path, output=True):
    # Initialize mediapipe solutions
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    frame_number = 0
    # Open video file
    cap = cv2.VideoCapture(video_path)

    filename = os.path.splitext(os.path.basename(video_path))[0]

    if output:
        # Prepare CSV and JSON files
        json_dir = os.path.join(os.path.dirname(video_path), 'json')  # JSON directory
        os.makedirs(json_dir, exist_ok=True)  # Create directory if it doesn't exist
        json_filename = os.path.join(json_dir, f'landmarks_{filename}.json')
        json_file = open(json_filename, 'w', encoding='UTF8')

        parquet_dir = os.path.join(os.path.dirname(video_path), 'parquet')  # Parquet directory
        os.makedirs(parquet_dir, exist_ok=True)  # Create directory if it doesn't exist
        parquet_filename = os.path.join(parquet_dir, f'landmarks_{filename}.parquet')

    json_data = []

    # Initialize mediapipe instances
    with mp_pose.Pose() as pose, mp_hands.Hands() as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and extract landmarks
            results_pose = pose.process(image)
            results_hands = hands.process(image)

            # Extract landmarks for pose, left hand, and right hand
            landmarks_pose = results_pose.pose_landmarks

            if results_hands.multi_hand_landmarks:
                # Check if there are any hand landmarks detected
                if len(results_hands.multi_hand_landmarks) == 1:
                    # Only one hand detected
                    landmarks_left_hand = results_hands.multi_hand_landmarks[0]
                    landmarks_right_hand = None
                elif len(results_hands.multi_hand_landmarks) == 2:
                    # Both hands detected
                    landmarks_left_hand = results_hands.multi_hand_landmarks[0]
                    landmarks_right_hand = results_hands.multi_hand_landmarks[1]
            else:
                # No hands detected
                landmarks_left_hand = None
                landmarks_right_hand = None

            serialized_pose = serialize_landmarks(landmarks_pose)
            serialized_left_hand = serialize_landmarks(landmarks_left_hand)
            serialized_right_hand = serialize_landmarks(landmarks_right_hand)

            # Create new index for each landmark type by enumerating
            for i, landmark in enumerate(serialized_pose):
                landmark['landmark_index'] = i
            for i, landmark in enumerate(serialized_left_hand):
                landmark['landmark_index'] = i
            for i, landmark in enumerate(serialized_right_hand):
                landmark['landmark_index'] = i

            # Write serialized landmarks to JSON
            json_data.append({
                'frame_number': frame_number,
                'pose': serialized_pose,
                'left_hand': serialized_left_hand,
                'right_hand': serialized_right_hand
            })

            frame_number += 1

    # Write JSON data to file
    json.dump(json_data, json_file, indent=4)

    # Convert JSON data to pandas DataFrame
    df = pd.json_normalize(json_data)

    # Expand the DataFrame format
    df = df.explode('pose')
    df = df.explode('left_hand')
    df = df.explode('right_hand')
    df = df.reset_index(drop=True)

    # Extract landmark data from the DataFrame
    landmark_data = df[['pose', 'left_hand', 'right_hand']]

    # Replace NaN values with None
    landmark_data = landmark_data.applymap(lambda x: x if pd.notna(x) else None)

    # Stack the landmark data into a single column
    landmark_data = landmark_data.stack().reset_index(level=1, drop=True)
    landmark_data.name = 'landmark'

    # Extract the landmark type, index, and coordinates
    df['landmark_type'] = landmark_data.apply(lambda x: x.get('type'))
    df['landmark_index'] = landmark_data.apply(lambda x: x.get('landmark_index'))
    df['x'] = landmark_data.apply(lambda x: x.get('x'))
    df['y'] = landmark_data.apply(lambda x: x.get('y'))
    df['z'] = landmark_data.apply(lambda x: x.get('z'))

    # Drop the original columns
    df.drop(['pose', 'left_hand', 'right_hand'], axis=1, inplace=True)

    df.drop(['pose', 'left_hand', 'right_hand'], axis=1, inplace=True)

    # Write DataFrame to parquet file
    if output:
        df.to_parquet(parquet_filename)

    # Close files
    json_file.close()

    # Close video file
    cap.release()

    # Print a success message
    print(f"Landmarks have been extracted and saved to JSON file {json_filename} and parquet file {parquet_filename}.")
    return df
