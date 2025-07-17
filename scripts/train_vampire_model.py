#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import vampire


treatment = ["24R_control", "24R_OGD", "2R_control", "2R_OGD", "30mR_control", "OGD_only", "ORST"]
groups = ['vampire_data']

image_set_path = "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data"

vampire.extraction.extract_properties(image_set_path)


build_info_df = pd.DataFrame({
    'img_set_path': [image_set_path],
    'output_path': [image_set_path],
    'model_name': ['li'],
    'num_points': [50],
    'num_clusters': [5],
    'num_pc': [np.nan]
})


vampire.quickstart.fit_models(build_info_df)


model_path = os.path.join('/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data', 'model_li_(50_5_39)__.pickle')
vampire_model = vampire.util.read_pickle(model_path)


main_path = "/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/testing/vampire_data"

apply_info_df = pd.DataFrame({
    'img_set_path': [
        f"{main_path}/24R_control",
        f"{main_path}/24R_OGD",
        f"{main_path}/2R_control",
        f"{main_path}/2R_OGD",
        f"{main_path}/30mR_control",
        f"{main_path}/OGD_ONLY",
        f"{main_path}/ORST",
    ],
    'model_path': ['/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle',
                   '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle',
                   '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle',
                   '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle',
                   '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle',
                   '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle',
                   '/Users/nelsschimek/Documents/nancelab/Data/mito_images/brendan_full_analysis/tifs/training/vampire_data/model_li_(50_5_39)__.pickle'],
    'output_path': [
        f"{main_path}/24R_control",
        f"{main_path}/24R_OGD",
        f"{main_path}/2R_control",
        f"{main_path}/2R_OGD",
        f"{main_path}/30mR_control",
        f"{main_path}/OGD_ONLY",
        f"{main_path}/ORST",
    ],
    'img_set_name': [
        "24R_control",
        "24R_OGD",
        "2R_control",
        "2R_OGD",
        "30mR_control",
        "OGD_ONLY",
        "ORST",
    ],
})


vampire.quickstart.transform_datasets(apply_info_df)

