
# %%
import napari
import numpy as np

viewer = napari.Viewer()
viewer.add_labels(name='test', 
                  data=np.array([[0, 1], [2, 3]]), 
                  colormap={0: 'transparent', 1: 'white', 2: 'red', 3: 'green'})

napari.run()
# %%
