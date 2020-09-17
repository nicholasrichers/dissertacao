def load_library(NAME_LIB, FILE_PATH):
    from importlib.machinery import SourceFileLoader
    FILE_PATH = '/content/dissertacao/' + FILE_PATH
    somemodule = SourceFileLoader(NAME_LIB, FILE_PATH).load_module()
    
    
def get_libs():
    #folder notebooks
    #load_library('paths', 'notebooks/Baseline/paths.py')

    
    #folder src.data
    load_library('make_dataset', 'src/data/make_dataset.py')


    #folder src.data
    load_library('train_model', 'src/models/train_model.py')
    load_library('evaluation', 'src/models/evaluation.py')

    #folder src.data
    load_library('metrics', 'src/validation/metrics.py')
    load_library('metrics_description', 'src/validation/metrics_description.py')

