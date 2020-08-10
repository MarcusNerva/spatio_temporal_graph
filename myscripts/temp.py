import numpy as np
import glob
import os


if __name__ == '__main__':
    spatial_path = '/Users/bismarck/Downloads/temp_data/object/spatial'
    temporal_path = '/Users/bismarck/Downloads/temp_data/object/temporal'

    spatial_list = glob.glob(os.path.join(spatial_path, '*.npy'))
    temporal_list = glob.glob(os.path.join(temporal_path, '*.npy'))

    print("here is spatial features:")
    for spatial in spatial_list:
        content = np.load(spatial)
        print(content.shape)
        print(content)

    # print("here is temporal features:")
    # for temporal in temporal_list:
    #     content = np.load(temporal)
    #     print(content.shape)
    #     print(content)

