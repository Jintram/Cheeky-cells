# %%

import glob
import os

# Functions/scripts that are only needed for specific cases


def anonimize_filenames(path):
    '''
    Go over files in "path", and rename them to "0001.extension" etc.
    '''
    
    
    folder='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheek-Cells_Anonymous/'
    
    allfiles = glob.glob(os.path.join(folder, '*'))
    
    for i, f in enumerate(allfiles):
        _, ext = os.path.splitext(f)
        newname = os.path.join(folder, f'{i:04d}{ext}')
        print(f'Renaming {f} to {newname}')
        os.rename(f, newname)