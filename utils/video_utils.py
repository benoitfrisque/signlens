import os
import json
import numpy as np
import mediapipe as mp
import cv2
import math
from google.protobuf.json_format import MessageToDict
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from signlens.params import N_LANDMARKS_HAND, N_LANDMARKS_POSE, LANDMARKS_VIDEO_DIR

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Constants for drawing landmarks on the image
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def serialize_landmarks(landmark_list):
    """
    Serialize a list of landmarks into a dictionary format.

    Args:
        landmark_list (list): A list of landmarks.

    Returns:
        list: A list of dictionaries, where each dictionary represents a landmark and contains the following keys:
            - 'landmark_index': The index of the landmark in the list.
            - 'x': The x-coordinate of the landmark. If the value is NaN, it is set to None.
            - 'y': The y-coordinate of the landmark. If the value is NaN, it is set to None.
            - 'z': The z-coordinate of the landmark. If the value is NaN, it is set to None.
    """
    landmarks = []
    for idx, landmark in enumerate(landmark_list.landmark):
        landmarks.append({
            'landmark_index': idx,
            'x': None if math.isnan(landmark.x) else landmark.x,
            'y': None if math.isnan(landmark.y) else landmark.y,
            'z': None if math.isnan(landmark.z) else landmark.z
        })
    return landmarks


def process_video_to_landmarks_json(video_path, output=True, show_preview=True, frame_interval=1, frame_limit=None, rear_camera=True, output_dir=LANDMARKS_VIDEO_DIR):
    """
    Process a video file and extract landmarks from each frame, then save the landmarks as JSON.
    Inspired from https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md

    Args:
        video_path (str): The path to the video file.
        output (bool, optional): Whether to save the landmarks as JSON. Defaults to True.
        show_preview (bool, optional): Whether to show a preview of the processed frames. Defaults to True.
        frame_interval (int, optional): The interval between processed frames. Defaults to 1.
        frame_limit (int, optional): The maximum number of frames to process. Defaults to None.
        rear_camera (bool, optional): Whether the video was recorded with a rear camera. Defaults to True.
        output_dir (str, optional): The directory to save the landmarks JSON file. Defaults to LANDMARKS_VIDEO_DIR.

    Returns:
        list: A list of dictionaries containing the extracted landmarks for each frame.

    Raises:
        FileNotFoundError: If the video file specified by `video_path` does not exist.

    Example:
        video_path = '/path/to/video.mp4'
        landmarks = process_video_to_landmarks_json(video_path, output=True, frame_interval=2, frame_limit=100)
        print(landmarks)
    """
    filename = os.path.splitext(os.path.basename(video_path))[0]

    # Open video file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' not found.")

    cap = cv2.VideoCapture(video_path)

    if output:
        # Prepare JSON file
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f'landmarks_{filename}.json')
        json_file = open(json_path, 'w', encoding='UTF8')

    json_data = []
    frame_number = 0
    processed_frames = 0

    # Initialize an empty NormalizedLandmarkList for hand and pose
    empty_hand_landmark_list = create_empty_landmark_list(N_LANDMARKS_HAND)
    empty_pose_landmark_list = create_empty_landmark_list(N_LANDMARKS_POSE)

    # Initialize mediapipe instances
    with mp_pose.Pose(static_image_mode=False) as pose, \
            mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Skip frames based on frame_interval
            if frame_number % frame_interval != 0:
                frame_number += 1
                continue

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and extract landmarks
            results_pose = pose.process(image_rgb)
            results_hands = hands.process(image_rgb)

            if show_preview:
            # Draw landmarks on the image
                annotated_image = draw_landmarks_on_image(image_rgb, results_hands, rear_camera)
                annotated_image_color = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imshow(f"Video {filename}", annotated_image_color)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Extract landmarks for pose, left hand, and right hand
            landmarks_pose = results_pose.pose_landmarks

            # Check if there are any pose landmarks detected
            if landmarks_pose is None:
                landmarks_pose = empty_pose_landmark_list

            # Initialize empty hand landmarkks, then overwrite if it finds it
            landmarks_left_hand = empty_hand_landmark_list
            landmarks_right_hand = empty_hand_landmark_list

            # Check if there are any hand landmarks detected
            if results_hands.multi_hand_landmarks:
                # Get handedness of each hand
                for idx, handedness in enumerate(results_hands.multi_handedness):
                    hand_side = get_hand_side(handedness, rear_camera)

                    if hand_side == 'left':
                        landmarks_left_hand = results_hands.multi_hand_landmarks[idx]
                    elif hand_side == 'right':
                        landmarks_left_hand = results_hands.multi_hand_landmarks[idx]

            serialized_pose = serialize_landmarks(landmarks_pose)
            serialized_left_hand = serialize_landmarks(landmarks_left_hand)
            serialized_right_hand = serialize_landmarks(landmarks_right_hand)

            # Write serialized landmarks to JSON
            json_data.append({
                'frame_number': frame_number,
                'pose': serialized_pose,
                'left_hand': serialized_left_hand,
                'right_hand': serialized_right_hand
            })

            frame_number += 1
            processed_frames += 1

            # Stop processing if frame_limit is reached
            if frame_limit is not None and processed_frames >= frame_limit:
                break

    # Close video file
    cap.release()

    if show_preview:
        cv2.destroyWindow(f"Video {filename}")

    if output:
        # Write JSON data to file
        json.dump(json_data, json_file, indent=4)
        # Close files
        json_file.close()
        print(f"✅ Landmarks saved to '{json_path}'")

    return json_data



def create_empty_landmark_list(n_landmarks):
    """
    Create an empty NormalizedLandmarkList.

    Args:
        n_landmarks (int): The number of landmarks to create.

    Returns:
        landmark_pb2.NormalizedLandmarkList: An empty NormalizedLandmarkList.

    """
    # Initialize an empty NormalizedLandmarkList for hand
    empty_landmark_list = landmark_pb2.NormalizedLandmarkList()

    # Add empty landmarks to the list
    for _ in range(n_landmarks):
        landmark = empty_landmark_list.landmark.add()
        landmark.x = np.nan  # We use nan and not None because it doesn't work with None
        landmark.y = np.nan
        landmark.z = np.nan

    return empty_landmark_list


def get_hand_side(handedness, rear_camera):
    """
    Determines the side of the hand based on the handedness classification.

    Args:
        handedness (protobuf message): The handedness classification message.
        rear_camera (bool): Flag indicating whether the input image is taken with a rear camera.

    Returns:
        str: The side of the hand ('left' or 'right').

    Notes:
        By default, mediapipe assumes the input image is mirrored, i.e., taken with a front-facing/selfie camera with images flipped horizontally.
        If you want to process images taken with a webcam/selfie, you can set rear_camera = False.
    """
    handedness_dict = MessageToDict(handedness)
    hand_side = handedness_dict['classification'][0]['label'].lower()

    if rear_camera:
        if hand_side == 'left':
            hand_side = 'right'
        elif hand_side == 'right':
            hand_side = 'left'

    return hand_side

def draw_landmarks_on_image(rgb_image, result_hands, rear_camera):
    """
    Draws landmarks on the given RGB image based on the detected hand landmarks.

    Args:
        rgb_image (numpy.ndarray): The RGB image on which to draw the landmarks.
        result_hands (mediapipe.python.solution_base.SolutionOutputs): The output of the hand detection model.
        rear_camera (bool): Flag indicating whether the camera is rear-facing or not.

    Returns:
        numpy.ndarray: The annotated image with landmarks drawn.

    Note:
        It is normal to see the left hand on the right side and the right hand on the left side if rear_camera=True.
    """
    annotated_image = np.copy(rgb_image)

    if result_hands.multi_hand_landmarks is None:
        return annotated_image

    # Loop through the detected hands to visualize.
    for idx in range(len(result_hands.multi_hand_landmarks)):
        hand_landmarks = result_hands.multi_hand_landmarks[idx].landmark
        handedness = result_hands.multi_handedness[idx]
        hand_side = get_hand_side(handedness, rear_camera)

        landmark_pb2.NormalizedLandmarkList()

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        # Draw the hand landmarks.
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{hand_side}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image
