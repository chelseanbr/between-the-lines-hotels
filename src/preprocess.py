import os
import pandas as pd

def merge_csv_mult_dir(path_to_dir):
    # Get all non-zip folders/files in dir
    names = os.listdir(path_to_dir)
    folders = [name for name in names if not name.endswith('.zip')] # Remove names found for zips
    print('Non-zip folders/files found in {}: {}'.format(path_to_dir, folders))
    # From each folder, read all CSV files
    dfs = []
    for folder in folders:
        try: 
            data_dir = path_to_dir + '/' + folder + '/'
            csv_filenames = os.listdir(data_dir)    
            for name in csv_filenames:
                df = pd.read_csv(data_dir + name)
                df['csv'] = name
                df['folder'] = folder
                dfs.append(df)
        except NotADirectoryError:
            print('\tSkipping (not a directory):', data_dir)
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all