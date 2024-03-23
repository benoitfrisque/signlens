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
	python -c 'from signlens.interface.main import train; train()'

#run_pred:
#python -c 'from signlens.interface.main import pred; pred()'

run_evaluate:
	python -c 'from signlens.interface.main import evaluate; evaluate()'

run_all: run_train run_evaluate

run_api:
	uvicorn signlens.api.fast:app --reload
