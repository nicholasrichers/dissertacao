def load_library(NAME_LIB, FILE_PATH, local=False):
    from importlib.machinery import SourceFileLoader

    if local==False:
        FILE_PATH = '/content/dissertacao/' + FILE_PATH

    else:
        FILE_PATH = '../../' + FILE_PATH


    somemodule = SourceFileLoader(NAME_LIB, FILE_PATH).load_module()
    





def get_libs(local=False):
    #folder notebooks
    #load_library('paths', 'notebooks/Baseline/paths.py') 

    
    #folder src.data
    load_library('make_dataset', 'src/data/make_dataset.py', local=local)


    #folder src.models
    load_library('train_model', 'src/models/train_model.py', local=local)
    load_library('evaluation', 'src/models/evaluation.py', local=local)
    load_library('meta_model', 'src/models/meta_model.py', local=local)
    load_library('neutralize', 'src/models/neutralize.py', local=local)
    load_library('get_model',  'src/models/get_model.py', local=local)
    load_library('xgb_ranker_class', 'src/models/xgb_ranker_class.py', local=local)
    load_library('pipelines', 'src/models/pipelines.py', local=local)



    #folder src.validation (MANTER A ORDEM)
    load_library('group_ts_split', 'src/validation/group_ts_split.py', local=local)
    load_library('combinatorial_split', 'src/validation/combinatorial_split.py', local=local)
    load_library('metrics_era', 'src/validation/metrics_era.py', local=local)
    load_library('metrics_description', 'src/validation/metrics_description.py', local=local)
    load_library('dsr', 'src/validation/dsr.py', local=local)
    load_library('metrics', 'src/validation/metrics.py', local=local)
    load_library('metrics_live', 'src/validation/metrics_live.py', local=local)
    load_library('stat', 'src/validation/stat.py', local=local)

    
    #folder src.visualization
    load_library('visualize', 'src/visualization/visualize.py', local=local)

    ################################################################
    ################################################################


    #folder src.collection/mlfinlab/cross_validation
    load_library('cross_validation', 'src/collection/mlfinlab/cross_validation/cross_validation.py', local=local)
    load_library('combinatorial', 'src/collection/mlfinlab/cross_validation/combinatorial.py', local=local)


    #folder src.collection/mlfinlab/feature_importance
    load_library('importance', 'src/collection/mlfinlab/feature_importance/importance.py', local=local)
    load_library('fingerpint', 'src/collection/mlfinlab/feature_importance/fingerpint.py', local=local)
    load_library('orthogonal', 'src/collection/mlfinlab/feature_importance/orthogonal.py', local=local)



    #folder src.collection/mlfinlab/clustering
    load_library('onc', 'src/collection/mlfinlab/clustering/onc.py', local=local)
    load_library('feature_clusters', 'src/collection/mlfinlab/clustering/feature_clusters.py', local=local)
    load_library('hierarchical_clustering', 'src/collection/mlfinlab/clustering/hierarchical_clustering.py', local=local)



    #folder src.collection/mlfinlab/codependence
    load_library('information', 'src/collection/mlfinlab/codependence/information.py', local=local)
    load_library('correlation', 'src/collection/mlfinlab/codependence/correlation.py', local=local)
    load_library('codependence_matrix', 'src/collection/mlfinlab/codependence/codependence_matrix.py', local=local)


    #folder src.collection/mlfinlab/util
    load_library('multiprocess', 'src/collection/mlfinlab/util/multiprocess.py', local=local)












