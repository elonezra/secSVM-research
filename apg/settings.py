# -*- coding: utf-8 -*-

"""
settings.py
~~~~~~~~~~~

Configuration options for the pipeline.

"""
import os

# The absolute path to the root folder of this project
_project_path = '/home/osboxes/sesvm/secscvm'
# The absolute path of the folder containing compiled Java components
_components_path = '/home/osboxes/sesvm/secscvm/java-components/build'

fromer_dataset="/home/osboxes/sesvm/secscvm/data_for_secsvm/former_malicious_test/0/"
datasets_for_secsvm="/home/osboxes/sesvm/secscvm/data_for_niv_avi/train/"
COM_dataset="/home/osboxes/sesvm/secscvm/data_for_secsvm/malicious_test/COM/COM4/results/"
SB_dataset="/home/osboxes/sesvm/secscvm/data_for_secsvm/malicious_test/SB/SB4/results/"
MB1_dataset="/home/osboxes/sesvm/secscvm/data_for_secsvm/malicious_test/MB1/MB1_4/results/"
MB2_dataset="/home/osboxes/sesvm/secscvm/data_for_secsvm/malicious_test/MB2/MB2_0/results/"
MB3_dataset="/home/osboxes/sesvm/secscvm/data_for_secsvm/malicious_test/MB3/MB3_4/results/"
MB4_dataset="/home/osboxes/sesvm/secscvm/data_for_niv_avi/test/"
def _project(base):
    return os.path.join(_project_path, base)


def _components(base):
    return os.path.join(_components_path, base)


config = {
    # Experiment settings
    'models': _project('data/models/'),
    'X_dataset': _project(datasets_for_secsvm+'train_dataset.json'),
    'y_dataset': _project(datasets_for_secsvm+'labels.json'),
    'X_dataset_test': _project(MB4_dataset+'test_dataset.json'),
    'y_dataset_test': _project(MB4_dataset+'labels.json'),
    #'meta': _project('data/features/apg-meta.json'),
    'indices': _project(''),  # only needed if using fixed indices
    # Java components
    'extractor': _components('extractor.jar'),
    'injector': _components('injector.jar'),
    'template_injector': _components('templateinjector.jar'),
    'cc_calculator': _components('cccalculator.jar'),
    'class_lister': _components('classlister.jar'),
    'classes_file': _project('all_classes.txt'),
    'extractor_timeout': 300,
    'cc_calculator_timeout': 600,
    # Other necessary components
    'android_sdk': '/usr/lib/android-sdk',
    'template_path': _project('template'),
    'mined_slices': _project('mined-slices'),
    'opaque_pred': _project('opaque-preds/sootOutput'),
    'resigner': _project('apk-signer.jar'),
    'feature_extractor': '/home/osboxes/sesvm/secscvm/feature-extractor',
    # Storage for generated bits-and-bobs
    'tmp_dir': '/home/osboxes/sesvm/secscvm/tmp',
    'ice_box': '/home/osboxes/sesvm/secscvm/ice_box',
    'results_dir': '/home/osboxes/sesvm/secscvm/res',
    'goodware_location': '/home/osboxes/sesvm/secscvm/apps/ben',
    'storage_radix': 0,  # Use if apps are stored with a radix (e.g., radix 3: root/0/0/A/00A384545.apk)
    # Miscellaneous options
    'tries': 1,
    'nprocs_preload': 8,
    'nprocs_evasion': 12,
    'nprocs_transplant': 8
}
