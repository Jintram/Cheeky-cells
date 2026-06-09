

from matplotlib.colors import ListedColormap, BoundaryNorm

# Bang wong color palette
color_palette_bangwong = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000"   # Black
]

# Bang wong again, but now black first (convenient for imgs with bg = 0)
color_palette_bangwong_blackfirst = [
    "#000000",   # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7"  # Reddish Purple    
]

# Bang wong again, but now white first (convenient for imgs with bg = 0)
color_palette_bangwong_whitefirst = [
    "#ffffff",   # Black
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7"  # Reddish Purple    
]

# convert any color list to a cmap
def colors_to_cmap(color_list, name_cmap='custom_cmap'):
        
    cmap = ListedColormap(color_list, name=name_cmap, N=len(color_list))
    
    norm  = BoundaryNorm(range(cmap.N + 1), cmap.N)
    
    return cmap, norm