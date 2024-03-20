from signlens.params import *

def model_save(model):
    model_path=os.path.join(BASE_DIR, 'utils','model.h5')
    model.save(model_path)
