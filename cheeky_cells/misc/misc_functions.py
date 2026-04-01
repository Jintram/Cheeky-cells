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
        
# %% ################################################################################
# Re-labeling code

def relabeling_oldcode(df_metadata):
    '''
    Custom function for when I removed a class.
    '''
    
    # Some custom code te re-map the stored seg files.
    # class 3 should be removed, and 4 should become 3, and 5 should become 4
    for file_idx in range(len(df_metadata)):
        # file_idx=0
        filename_base = os.path.splitext(df_metadata.loc[file_idx, 'filename'])[0]
        segfile = segfolder + filename_base + '_tile_seg.npy'
        current_seg = np.load(segfile, allow_pickle=True)
        # remap
        current_seg_remapped = current_seg.copy()
        current_seg_remapped[current_seg==5] = 4
        current_seg_remapped[current_seg==4] = 3
        current_seg_remapped[current_seg==3] = 2
        # save back
        np.save(segfile, current_seg_remapped)