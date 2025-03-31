


import napari
import numpy as np
from PIL import Image


viewer = napari.Viewer()

@viewer.bind_key('t') # test function
def test_fn(viewer):
    print('just testing')

filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized/resizedTo20_Cellen tellen, 40x foto 4.jpg'
img_PIL = Image.open(filepath)
img = np.array(img_PIL)
viewer.add_image(img)

drawing_active = False  # Toggle state
current_polygon = []  # Stores current polygon points
min_distance = 5  # Minimum pixel distance to add a new point

def toggle_drawing(viewer):
    global drawing_active, current_polygon
    
    drawing_active = not drawing_active  # Toggle state
    
    if drawing_active:
        print("Drawing mode activated")   
        # Register the callback
        viewer.mouse_move_callbacks.append(print_mouse_position)     
    else:
        print("Drawing mode deactivated")
        viewer.mouse_move_callbacks.remove(print_mouse_position)

# Function to print mouse position continuously
def print_mouse_position(viewer, event):    
    """Prints the mouse position coordinates whenever the mouse moves."""
    global drawing_active, current_polygon, min_distance
    
    pos = event.position
    if pos is not None:
        print(f"Mouse position: {pos[:2]}")

@viewer.bind_key('p')
def toggle_drawing_key(viewer):
    print('test p')
    toggle_drawing(viewer)#viewer)

@viewer.bind_key('m')    
def engage_mouse(viewer):
    viewer.mouse_move_callbacks.append(print_mouse_position)     


# Add an empty shapes layer for the polygon
# polygon_layer = viewer.add_shapes(shape_type='polygon', edge_color='red', face_color='transparent')

napari.run()