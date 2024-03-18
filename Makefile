install_requirements:
	@pip install --upgrade pip
	@pip install -r requirements.txt
	@sudo apt-get install ffmpeg

create_virtual_env:
	@pip install --upgrade pip
	@pyenv virtualenv 3.10.6 signlens
	@pyenv local signlens
