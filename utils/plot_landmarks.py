import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.express as px
from ipywidgets import interact, widgets
from IPython.display import HTML

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
    if landmark_type is None:
        return None
    if 'hand' in landmark_type.lower():
        return mp_hands.HAND_CONNECTIONS
    if landmark_type.lower() == 'pose':
        return mp_pose.POSE_CONNECTIONS

    return None


def plot_landmarks_2D_from_dict(landmarks, landmark_type=None, lm_color=None, lm_size=10):
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

def plot_interactive_sequences_2D_from_dict(landmarks_dict):
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
            plot_landmarks_2D_from_dict(landmarks_sequence[frame_index], landmark_type)

        plt.xlim(-0.5, 1.5)
        plt.ylim(-2.5, 0)
        plt.title(f"Frame {frame_index:03d}")

    return interact(plot_sequences_2D, frame_index=widgets.SelectionSlider(description='Frame index', continuous_update=True,  options=range(0, n_frames)))

def video_2D_from_dict(landmarks_dict, figsize=None, repeat=True, interval=200):
    """
    Create an interactive 2D video from a dictionary of landmarks.

    Args:
        landmarks_dict (dict): A dictionary containing sequences of landmarks for different types.
        figsize (tuple): Figure size.
        repeat (bool): Whether to repeat the animation.
        interval (int): Interval between frames in milliseconds.

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
            plot_landmarks_2D_from_dict(landmarks_sequence[frame_index], landmark_type)

        plt.xlim(-0.5, 1.5)
        plt.ylim(-2.5, 0)
        plt.title(f"Frame {frame_index:03d}")

    fig = plt.figure(figsize=figsize)
    animation = FuncAnimation(fig, plot_sequences_2D, frames=n_frames, interval=interval, repeat=repeat)
    video = HTML(animation.to_html5_video())
    plt.close(fig)
    return video

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
    # Transform coordinates: -z, x, -y
    plotted_landmarks = np.array([-landmarks[:, 2], landmarks[:, 0], -landmarks[:, 1]]).T

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


def plot_landmarks_3D_interactive(landmarks, landmark_type=None, fig=None):
    """
    Plot 3D landmarks interactively.

    Args:
        landmarks (numpy.ndarray): Array of shape (N, 3) representing the 3D coordinates of landmarks.
        landmark_type (str or None): Type of landmarks.
        fig (plotly.graph_objects.Figure): Existing figure.

    Returns:
        plotly.graph_objects.Figure: Interactive plotly figure.
    """
    # Transform coordinates: -z, x, -y
    plotted_landmarks = np.array([-landmarks[:, 2], landmarks[:, 0], -landmarks[:, 1]]).T

    if fig is None:
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


# Load from file instead of dict
def plot_interactive_sequences_2D_from_pq_file(pq_file_path, *args, **kwargs):
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
    return plot_interactive_sequences_2D_from_dict(landmarks_dict, *args, **kwargs)

def video_2D_from_pq_file(pq_file_path, *args, **kwargs):
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
    return video_2D_from_dict(landmarks_dict, *args, **kwargs)

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
