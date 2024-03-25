install_requirements:
	@pip install --upgrade pip
	@pip install -r requirements.txt

install_signlens_dev:
	@pip install -e .

install_requirements_dev:
	@make -s install_requirements
	@pip install -r requirements_dev.txt
	@if [ $$(uname) = "Darwin" ]; then \
				brew install ffmpeg; \
    else \
        sudo apt-get install ffmpeg; \
    fi
	@make -s install_signlens_dev

create_virtual_env:
	@pip install --upgrade pip
	@pyenv virtualenv 3.10.6 signlens
	@pyenv local signlens

create_output_dir:
	@mkdir -p training_outputs

reset_output_dir:
	@rm -rf training_outputs
	@make -s create_output_dir

run_preprocess:
	python -c 'from signlens.interface.main import preprocess; preprocess()'

run_train:
	@make -s create_output_dir
	python -c 'from signlens.interface.main import train; train()'

#run_pred:
#python -c 'from signlens.interface.main import pred; pred()'

# Use 'make run_evaluate' or 'run_evaluate 42' to run the evaluation (42 is a random_state)
run_evaluate:
	@python -c 'from signlens.interface.main import evaluate; evaluate($(filter-out $@,$(MAKECMDGOALS)))'
%:
	@:

# Use 'make run_all' to run the whole pipeline or 'make run_all 42' to run the whole pipeline with a random_state
run_all:
	@make -s create_output_dir
	python -c 'from signlens.interface.main import main; main($(filter-out $@,$(MAKECMDGOALS)))'
%:
	@:

run_api:
	uvicorn signlens.api.fast:app --reload

run_docker:
	docker run -it \
	-e NUM_CLASSES=250 \
	-e MAX_SEQ_LEN=100 \
	-e DATA_FRAC=1 \
	-e PORT=8000 -p 8000:8000 ${GAR_IMAGE}:dev

run_docker_prod:
	docker run -it \
	-e NUM_CLASSES=250 \
	-e MAX_SEQ_LEN=100 \
	-e DATA_FRAC=1 \
	-e PORT=8000 -p 8000:8000 ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/signlens/${GAR_IMAGE}:prod

build_docker:
	docker build -t ${GAR_IMAGE}:dev .

build_docker_prod:
	docker build -t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/signlens/${GAR_IMAGE}:prod .


google_configure:
	gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
	gcloud artifacts repositories create signlens --repository-format=docker \
--location=${GCP_REGION} --description="Repository for Signlens - sign language classification"

google_push:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/signlens/${GAR_IMAGE}:prod
