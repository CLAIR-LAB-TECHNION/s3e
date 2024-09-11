import glob
import json
import os

import pandas as pd
from tqdm.auto import tqdm


def get_cooccurrence_matrix(states_data_dir):
    # load all state dictionaries
    discovered_states = {}
    for fname in tqdm(glob.glob(os.path.join(states_data_dir, '*.json')), desc='loading co-occurrence data'):
        with open(fname, 'r') as f:
            discovered_states[fname] = json.load(f)
    
    # save as a table. rows are datapoints, columns are predicates
    df = pd.DataFrame.from_dict(discovered_states).T

    # drop duplicates and 
    df.drop_duplicates(inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)

    # calculate boolean coocurrence matrix
    return df.T @ df

