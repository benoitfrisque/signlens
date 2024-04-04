# SignLens - Deep Learning for Sign Language Classification

SignLens is a project that leverages the power of Deep Learning to classify American Sign Language (ASL) gestures. The project includes scripts for data preprocessing, model training, and prediction. It is a final project for the Data Science bootcamp at Le Wagon.


## Table of Contents
- [Data](#data)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [Contact](#contact)


## Data
The data used in this project comes from the following sources:
- [ASL Signs Kaggle Competition](https://www.kaggle.com/competitions/asl-signs)
- [WLASL Processed Dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

Please download the 2 folders and put them in a folder called `raw_data`. The structure of this folder should look like this:
```
raw_data
├── asl-signs
│   ├── sign_to_prediction_index_map.json
│   ├── train.csv
│   ├── train_landmark_files
├── WLASL
│   ├── missing.txt
│   ├── nslt_1000.json
│   ├── nslt_100.json
│   ├── nslt_2000.json
│   ├── nslt_300.json
│   ├── videos
│   ├── wlasl_class_list.txt
│   └── WLASL_v0.3.json
```
If you don't have enough storage on your computer, you can keep the asl-signs directory only.

The asl-signs directory is still quite heavy. As for this project, the face landmarks are not being used, you can process the parquet files and remove the face landmarks by running the following command:
```bash
python python signlens/preprocessing/preprocess_files_light.py
```

This will generate lighter parquet files that only contain relevant landmarks. You will then have a folder called train_landmark_files_no_face and you can get rid of the train_landmark_files.

The WLASL directory is only used for prediction on videos. It is not used to train the model.


## Getting Started

<details>
  <summary>Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.</summary>


### Prerequisites

- Python 3.7 or later (Python 3.10.6 is recommended)
- TensorFlow
- OpenCV
- Pandas
- Numpy

### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/benoitfrisque/signlens.git
```

2. Navigate into the project directory
```bash
cd signlens
```

3. Create a local environment of Python 3.10.6 with pyenv. This requires you to have pyenv installed.
```bash
make create_virtual_env
```

4. Install the required Python libraries:
```bash
make install_requirements_dev
```

This will install all the libraries from requirements.txt and requirements_dev.txt.
</details>

## Usage

### Model training

1. Copy the .env.sample into .env, and adapt the parameters accordingly. We recommend using a Virtual Machine with large RAM size to handle fraction of the data (frac) > 0.5.

2. Train the model with the following command
```bash
make run_all
```


### Prediction
Here is an example of how to use the trained model for prediction:

```python
from signlens import Model

# Load the model
model = Model.load('path-to-model-file')

# Predict the class of a sign
sign = 'path-to-sign-image'
print(model.predict(sign))
```

### API On Google Cloud
<details>
  <summary>Expand for more details</summary>

#### Prerequisites:
1. **Google Cloud Platform Account**: You need an active Google Cloud Platform (GCP) account.
2. **Docker**: Ensure Docker is installed on your local machine for building containerized applications.
3. **Google Cloud SDK**: Install the Google Cloud SDK to interact with GCP services via the command line.
4. **Google Cloud Project**: Create a new or select an existing Google Cloud project where you'll deploy the API.
5. **Billing Enabled**: Ensure billing is enabled for your Google Cloud project.

#### Dependencies:
1. **FastAPI**: Install the FastAPI library, a modern, fast (high-performance), web framework for building APIs with Python 3.6+.
2. **uvicorn**: Install uvicorn, an ASGI server implementation.
3. **Dockerfile**: Create a Dockerfile for containerizing the API application.
4. **Google Cloud Build**: Enable Google Cloud Build for automating builds and deployments.

#### Steps for Deployment:

1. **Test the API Locally**:
   - Run the API locally using FastAPI and verify that it works as expected.
```bash
make run_api
```

2. **Build Docker Image Locally**:
```bash
make build_docker
```

3. **Test Docker Image Locally**:
   - Run the Docker container locally to ensure it behaves correctly. Upload a Json file of preprocessed landmarks and checks thatthe prediction is correct.7

4. **Build Production Docker Image**
    - If it worked correctly, build the production image:
```bash
make build_docker_prod
```

5. **Push Image to Google Artifact Registry**
    - To deploy it on Google Cloud, copy `.env.sample.yaml` and rename it `.env.yaml`.
    - Adapt your Google region, project_ID ect in `.env.yaml` and `.env`.
    - Run the following command to push the prod image:
```bash
make google_push
```

6. **Deploy API to Google Cloud Run**:
   - Deploy the Docker image from GAR to Google Cloud Run using the gcloud command-line tool.
```bash
make google_deploy
```

7. **Verify Deployment**:
   - Verify that the API is successfully deployed and accessible via the provided URL.

8. **Configure Domain (Optional)**:
    - If needed, configure a custom domain for your API endpoint.

9. **Monitor and Maintain**:
    - Monitor the deployed API for performance, errors, and usage.
    - Regularly update and maintain the API as needed.

</details>

## Authors

- [Benoît Frisque](https://github.com/benoitfrisque)
- [Wail Benrabh](https://github.com/WailBen97)
- [Jan Storz](https://github.com/janstorz)
- [Maximilien Grieb](https://github.com/MaxGrieb)

## Contact

If you have any questions, comments, or feedback, feel free to reach out to us!

- Email: benoitfrisque@gmail.com
- GitHub: [benoitfrisque](https://github.com/benoitfrisque)
