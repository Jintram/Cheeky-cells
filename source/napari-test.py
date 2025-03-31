

# This was very instructive, but also very useless, 
# as there is already an option to make a "lasso", 
# you just have to have a polygon layer, and use 
# the lasso tool (SHIFT + P).



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

polygon_layer = viewer.add_shapes(shape_type='polygon', edge_color='red', face_color='transparent', name='drawlayer')
polygon_layer_committed = viewer.add_shapes(shape_type='polygon', edge_color='red', face_color='transparent', name='committedlayer')

drawing_active = False  # Toggle state
current_polygon = []  # Stores current polygon points
min_distance = 5  # Minimum pixel distance to add a new point

# Function to print mouse position continuously
def print_mouse_position(viewer, event):    
    """Prints the mouse position coordinates whenever the mouse moves."""
    global drawing_active, current_polygon, min_distance
    
    pos = event.position
    if pos is not None:
        print(f"Mouse position: {pos[:2]}")
    
    if (len(current_polygon)==0):
        current_polygon = [pos]
    elif(np.linalg.norm(np.array(pos) - np.array(current_polygon[-1])) > min_distance):
        current_polygon.append(pos)
        polygon_layer.data = [np.array(current_polygon)]  # Update polygon

def toggle_drawing(viewer):
    global drawing_active, current_polygon
    
    drawing_active = not drawing_active  # Toggle state
    
    if drawing_active:
        print("Drawing mode activated")   
        # Register the callback
        viewer.mouse_move_callbacks.append(print_mouse_position) 
    else:
        print("Drawing mode deactivated")
        current_polygon=[]            
        polygon_layer_committed.data = polygon_layer_committed.data + polygon_layer.data                
        viewer.mouse_move_callbacks.remove(print_mouse_position)

print('1')
@viewer.bind_key('p', overwrite=True)
def toggle_drawing_key(viewer):
    print('test p')
    # toggle_drawing(viewer)#viewer)

print('2')
@viewer.bind_key('m', overwrite=True)
def engage_mouse(viewer):
    viewer.mouse_move_callbacks.append(print_mouse_position)     

print('3')
@viewer.bind_key('k', overwrite=True)
def another_test(viewer):
    print('(De)activating custom polygon.')
    toggle_drawing(viewer)

print('4')
@viewer.bind_key('t', overwrite=True)
def another_test(viewer):
    print('test 123')

# Add an empty shapes layer for the polygon
# polygon_layer = viewer.add_shapes(shape_type='polygon', edge_color='red', face_color='transparent')

napari.run()