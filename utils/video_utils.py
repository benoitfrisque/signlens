import os
#import queue
import json
import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import av
#import pyarrow.csv as pv
#import pyarrow.parquet as pq

################################################################################
# Extract Landmarks to file and dataframe
################################################################################

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

def serialize_landmark(frame_number, landmark_type, landmark_index, landmark):
    '''
    Serialize a single landmark into a dictionary format, specifically for parquet compatibility.

    Args:
        frame_number (int): The frame number of the landmark.
        landmark_type (str): The type of the landmark (e.g., "right_hand", "left_hand", "face").
        landmark_index (int): The index of the landmark within the given type.
        landmark (mediapipe.python.solution_base.Landmark): The landmark object.

    Returns:
        dict: A dictionary representing the serialized landmark. It contains the following keys:
            - 'frame': The frame number of the landmark.
            - 'row_id': A unique identifier for the landmark consisting of the frame number, landmark type, and landmark index.
            - 'type': The type of the landmark.
            - 'landmark_index': The index of the landmark within the given type.
            - 'x': The x-coordinate of the landmark.
            - 'y': The y-coordinate of the landmark.
            - 'z': The z-coordinate of the landmark.
    '''
    return {
        "frame": frame_number,
        "row_id": f"{frame_number}-{landmark_type}-{landmark_index}",
        "type": landmark_type,
        "landmark_index": landmark_index,
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z
    }

def process_video_to_landmarks(video_path, output=True):
    '''
    Extract landmarks from a video file and save them to a JSON file and a parquet file.

    Args:
        video_path (str): The path of the video file.
        output (bool): Whether to save the extracted landmarks to files (default is True).

    Returns:
        DataFrame: A pandas DataFrame containing the extracted landmarks.
    '''
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
                #'row_id': row_id,
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

       # magic columns
    df['type'] = df.apply(lambda x: 'pose' if pd.notna(x['pose']) else ('left_hand' if pd.notna(x['left_hand']) else 'right_hand'), axis=1)
    df['landmark_index'] = df.apply(lambda x: x['pose'].get('landmark_index') if pd.notna(x['pose']) else (x['left_hand'].get('landmark_index') if pd.notna(x['left_hand']) else x['right_hand'].get('landmark_index')), axis=1)
    df['row_id'] = df.apply(lambda x: f"{x['frame_number']}-{x['type']}-{x['landmark_index']}", axis=1)
    df['x'] = df.apply(lambda x: x['pose'].get('x') if pd.notna(x['pose']) else (x['left_hand'].get('x') if pd.notna(x['left_hand']) else x['right_hand'].get('x')), axis=1)
    df['y'] = df.apply(lambda x: x['pose'].get('y') if pd.notna(x['pose']) else (x['left_hand'].get('y') if pd.notna(x['left_hand']) else x['right_hand'].get('y')), axis=1)
    df['z'] = df.apply(lambda x: x['pose'].get('z') if pd.notna(x['pose']) else (x['left_hand'].get('z') if pd.notna(x['left_hand']) else x['right_hand'].get('z')), axis=1)

    # Drop the original columns
    df.drop(['pose', 'left_hand', 'right_hand'], axis=1, inplace=True)

    # Write DataFrame to parquet file
    if output:
        df.to_parquet(parquet_filename)

    # Close files
    json_file.close()

    # Close video file
    cap.release()

    # Print a success message
    print(f"Landmarks have been extracted and saved to JSON file {json_filename} and parquet file {parquet_filename} .")
    return df

################################################################################
# Live Video buffering
################################################################################

def frame_buffer_callback(frame, frame_buffer):
    '''
    Update a FIFO queue with the newest frame.

    Usage: webrtc_streamer(key="example", video_frame_callback=frame_buffer_callback)

    Args:
        - current frame (av.VideoFrame)
        - frame_buffer (queue.Queue): A FIFO queue to store the frames

    Returns:
        - transformed frame (av.VideoFrame)
        - updated frame buffer (queue.Queue)
        - DataFrame containing the extracted landmarks
    '''
    #frame_buffer = queue.Queue(maxsize=10)
    img = frame.to_ndarray(format="bgr24")

    # Ensure the buffer is full with 10 frames
    if frame_buffer.full():
        frame_buffer.get()  # Remove the oldest frame if the buffer is full
    frame_buffer.put(img)  # Add the current frame to the buffer

    # Process the frames
    df = process_video_to_landmarks(frame_buffer, output=False)

    return av.VideoFrame.from_ndarray(img, format="bgr24"), frame_buffer, df
