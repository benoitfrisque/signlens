import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.express as px
from ipywidgets import interact, widgets, Output, GridspecLayout
from IPython.display import HTML
from IPython import display

# Importing functions for preprocessing and loading data
from signlens.preprocessing.preprocess_files import load_relevant_data_subset_per_landmark_type

# Importing modules for landmark detection
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import pose as mp_pose

def get_connections(landmark_type):
    """
    Get the connections between landmarks based on the landmark type.

    Args:
        landmark_type (str or None): Type of landmarks.

    Returns:
        list or None: List of landmark connections.
    """
    if not landmark_type:
        return None
    if 'hand' in landmark_type.lower():
        return mp_hands.HAND_CONNECTIONS
    if landmark_type.lower() == 'pose':
        return mp_pose.POSE_CONNECTIONS

    return None

def plot_landmarks_2D(landmarks, landmark_type=None, lm_color=None, lm_size=10):
    """
    Draw 2D landmarks on a matplotlib plot.

    Args:
        landmarks (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates of landmarks.
        landmark_type (str or None): Type of landmarks to draw.
        lm_color (str or None): Color of the landmarks.
        lm_size (float): Size of the landmarks.

    Raises:
        ValueError: If the landmark index is out of range or if the connection is invalid.

    Returns:
        None
    """
    connections = get_connections(landmark_type)

    # Transform coordinates: x, -y
    plotted_landmarks = np.array([landmarks[:, 0], -landmarks[:, 1]]).T

    plt.scatter(
        x=plotted_landmarks[:, 0],
        y=plotted_landmarks[:, 1],
        color=lm_color,
        s=lm_size
    )

    if connections:
        num_landmarks = landmarks.shape[0]
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            landmark_pair = np.array([plotted_landmarks[start_idx, :], plotted_landmarks[end_idx, :]])
            plt.plot(
                landmark_pair[:, 0],
                landmark_pair[:, 1],
                color='black',
                linewidth=0.5
            )

def plot_interactive_landmark_frames_2D_from_dict(landmarks_dict, title=None):
    """
    Plot interactive 2D sequences of landmarks.

    Args:
        landmarks_dict (dict): A dictionary containing sequences of landmarks for different types.

    Returns:
        function: A function for interactive plotting.
    """
    n_frames = len(landmarks_dict[next(iter(landmarks_dict))])

    def plot_sequences_2D(frame_index):
        """
        Plot 2D sequences of landmarks for a given frame index.

        Args:
            frame_index (int): Index of the frame to be plotted.
        """
        ax = plt.gca()
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        for landmark_type, landmarks_sequence in landmarks_dict.items():
            plot_landmarks_2D(landmarks_sequence[frame_index], landmark_type)

        plt.xlim(-0.5, 1.5)
        plt.ylim(-2.5, 0)

        if title:
            plt.title(f"{title}\nFrame {frame_index:03d}")
        else:
            plt.title(f"Frame {frame_index:03d}")

    return interact(plot_sequences_2D, frame_index=widgets.SelectionSlider(description='Frame index', continuous_update=True,  options=range(0, n_frames)))

def video_landmarks_2D_from_dict(landmarks_dict, figsize=None, title=None, repeat=True, interval=200, show_axes=True):
    """
    Create an interactive 2D video from a dictionary of landmarks.

    Args:
        landmarks_dict (dict): A dictionary containing sequences of landmarks for different types.
        figsize (tuple, optional): Figure size. Default is None.
        title (str, optional): Title for the video. Default is None.
        repeat (bool, optional): Whether to repeat the animation. Default is True.
        interval (int, optional): Interval between frames in milliseconds. Default is 200.
        show_axes (bool, optional): Whether to show axes on the plot. Default is True.

    Returns:
        HTML: Interactive HTML video.
    """
    n_frames = len(landmarks_dict[next(iter(landmarks_dict))])

    def plot_sequences_2D(frame_index):
        """
        Plot 2D sequences of landmarks for a given frame index.

        Args:
            frame_index (int): Index of the frame to be plotted.
        """
        ax = plt.gca()
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        for landmark_type, landmarks_sequence in landmarks_dict.items():
            plot_landmarks_2D(landmarks_sequence[frame_index], landmark_type)

        if not show_axes:
            ax.axis('off')

        plt.xlim(-0.5, 1.5)
        plt.ylim(-2.5, 0)
        if title:
            plt.title(f"{title}\nFrame {frame_index:03d}")
        else:
            plt.title(f"Frame {frame_index:03d}")

    fig = plt.figure(figsize=figsize)
    animation = FuncAnimation(fig, plot_sequences_2D, frames=n_frames, interval=interval, repeat=repeat)
    video = HTML(animation.to_html5_video())
    plt.close(fig)
    return video

def video_grid_landmarks_2D(train, sign, n_videos, shuffle=True, num_columns=5, interval=100, figsize=(2,3), show_axes=False):
    """
    Create a grid layout of 2D landmark videos for a specified sign from a subset of training data.

    Args:
        train (pandas.DataFrame): DataFrame containing training data.
        sign (str): Sign for which the videos are to be displayed.
        n_videos (int): Number of videos to display.
        shuffle (bool, optional): Whether to shuffle the videos before selecting. Defaults to True.
        num_columns (int, optional): Number of columns in the grid layout. Defaults to 5.
        interval (int, optional): Interval between frames in milliseconds. Defaults to 100.
        figsize (tuple, optional): Figure size for each video. Defaults to (2, 3).
        show_axes (bool, optional): Whether to show axes on the plot. Default is False.

    Returns:
        ipywidgets.GridspecLayout: Grid layout containing the displayed videos.
    """
    # Check if the specified sign exists in the training data
    if sign not in train['sign'].unique():
        raise ValueError(f"The sign '{sign}' does not exist in the training data.")

    # Select subset of videos for the selected sign
    train_sign = train[train['sign'] == sign]
    if shuffle:
        train_sign = train_sign.sample(n_videos).reset_index()
    else:
        train_sign = train_sign.iloc[:n_videos].reset_index()

    # Create a new grid layout
    num_rows = math.ceil(n_videos / num_columns)
    grid = GridspecLayout(num_rows, num_columns)

    # Iterate over the file paths and stop when the maximum number of videos is reached
    for i, sequence in train_sign.iterrows():
        # Calculate the row and column indices for placing the video in the grid
        row_index = i // num_columns
        column_index = i % num_columns

        # If the maximum number of videos is reached, break out of the loop
        if i >= num_rows * num_columns:
            break

        # Output widget for the current video
        out = Output()
        with out:
            video = video_landmarks_2D_from_pq_file(sequence['file_path'], interval=interval, figsize=figsize, title=sequence['sequence_id'], show_axes=show_axes)
            display.display(video)
        grid[row_index, column_index] = out

    print("-"*20)
    print(f"{sign}")
    print("-"*20)
    return grid


def plot_landmarks_3D(ax, landmarks, landmark_type=None, connections=None, lm_color=None, lm_size=10):
    """
    Plot 3D landmarks on a matplotlib 3D plot.

    Args:
        ax: Matplotlib 3D axis.
        landmarks (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates of landmarks.
        landmark_type (str or None): Type of landmarks to draw.
        connections (list or None): List of landmark connections.
        lm_color (str or None): Color of the landmarks.
        lm_size (float): Size of the landmarks.

    Raises:
        ValueError: If the landmark index is out of range or if the connection is invalid.

    Returns:
        None
    """

    plotted_landmarks = np.array([-landmarks[:, 2], landmarks[:, 0], -landmarks[:, 1]]).T # Transform coordinates: -z, x, -y

    ax.scatter3D(
        xs=plotted_landmarks[:, 0],
        ys=plotted_landmarks[:, 1],
        zs=plotted_landmarks[:, 2],
        color=lm_color,
        s=lm_size
    )

    connections = get_connections(landmark_type)

    if connections:
        num_landmarks = landmarks.shape[0]
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            landmark_pair = np.array([plotted_landmarks[start_idx, :], plotted_landmarks[end_idx, :]])
            ax.plot3D(
                xs=landmark_pair[:, 0],
                ys=landmark_pair[:, 1],
                zs=landmark_pair[:, 2],
                color='black',
                linewidth=0.5
            )

def plot_all_landmark_types_3D_from_dict(landmarks_dict, frame_index):
    """
    Plot 3D landmarks for all landmark types from a dictionary.

    Args:
        landmarks_dict (dict): A dictionary containing sequences of landmarks for different types.
        frame_index (int): Index of the frame to be plotted.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10,5))

    for i, (landmark_type, landmarks_sequence) in enumerate(landmarks_dict.items(), start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        plot_landmarks_3D(ax, landmarks_sequence[frame_index], landmark_type)
        plt.title(landmark_type)

def plotly_landmarks_3D(landmarks, landmark_type=None, fig=None):
    """
    Plot 3D landmarks interactively using plotly.express.

    Args:
        landmarks (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates of landmarks.
        landmark_type (str or None): Type of landmarks.
        fig (plotly.graph_objects.Figure): Existing figure.

    Returns:
        plotly.graph_objects.Figure: Interactive plotly figure.
    """
    # Transform coordinates: -z, x, -y
    plotted_landmarks = np.array([-landmarks[:, 2], landmarks[:, 0], -landmarks[:, 1]]).T

    if not fig:
        # Scatter plot for landmarks
        fig = px.scatter_3d(
            x=plotted_landmarks[:, 0],
            y=plotted_landmarks[:, 1],
            z=plotted_landmarks[:, 2]
        )
    else:
        fig.add_trace(px.scatter_3d(
            x=plotted_landmarks[:, 0],
            y=plotted_landmarks[:, 1],
            z=plotted_landmarks[:, 2]
        ).data[0])  # Adding trace to the figure

    connections = get_connections(landmark_type)

    if connections:
        num_landmarks = landmarks.shape[0]
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                 f'from landmark #{start_idx} to landmark #{end_idx}.')
            landmark_pair = np.array([plotted_landmarks[start_idx, :], plotted_landmarks[end_idx, :]])
            fig.add_trace(px.line_3d(
                x=landmark_pair[:, 0],
                y=landmark_pair[:, 1],
                z=landmark_pair[:, 2],
            ).data[0])  # Adding trace to the figure

    return fig


################################################################################
# Load from file instead of dict
################################################################################
def plot_interactive_landmark_frames_2D_from_pq_file(pq_file_path, *args, **kwargs):
    """
    Plot interactive 2D sequences of landmarks from a Parquet file.

    Args:
        pq_file_path (str): Path to the Parquet file.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        function: A function for interactive plotting.
    """
    landmarks_dict = load_relevant_data_subset_per_landmark_type(pq_file_path)
    plot_interactive_landmark_frames_2D_from_dict(landmarks_dict, *args, **kwargs)

def video_landmarks_2D_from_pq_file(pq_file_path, *args, **kwargs):
    """
    Create an interactive 2D video from a Parquet file.

    Args:
        pq_file_path (str): Path to the Parquet file.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        HTML: Interactive HTML video.
    """
    landmarks_dict = load_relevant_data_subset_per_landmark_type(pq_file_path)
    return video_landmarks_2D_from_dict(landmarks_dict, *args, **kwargs)

def plot_all_landmark_types_3D_from_pq_file(pq_file_path, frame_index, *args, **kwargs):
    """
    Plot 3D landmarks for all landmark types from a Parquet file.

    Args:
        pq_file_path (str): Path to the Parquet file.
        frame_index (int): Index of the frame to be plotted.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        None
    """
    landmarks_dict = load_relevant_data_subset_per_landmark_type(pq_file_path)
    plot_all_landmark_types_3D_from_dict(landmarks_dict, frame_index)
