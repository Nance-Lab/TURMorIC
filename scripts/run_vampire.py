import os
import vampire
from turmoric.utils import create_metadata_df_from_json


def run_vampire(metadata_json):

    #training_df = create_metadata_df_from_json(metadata_json)
    #vampire.quickstart.fit_models(training_df)

    #model_path = os.path.join(metadata_json['base_dir'], metadata_json['model_name'])
    #vampire_model = vampire.util.read_pickle(model_path)

    testing_df = create_metadata_df_from_json(metadata_json, mode='testing')
    vampire.quickstart.transform_datasets(testing_df)

    return

# Example usage
if __name__ == "__main__":
    run_vampire('/Users/nelsschimek/Documents/nancelab/software_packages/turmoric/vampire_metdata.json')