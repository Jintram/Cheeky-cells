



from PIL import Image

import napari
import numpy as np

filepath='/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized/resizedTo20_Cellen tellen, 40x foto 4.jpg'

# load the image in filepath using PIL
img_PIL = Image.open(filepath)
# convert the image to a np.array
img = np.array(img_PIL)

###

# Parameters
drawing_active = False  # Toggle state
current_polygon = []  # Stores current polygon points
min_distance = 5  # Minimum pixel distance to add a new point

def on_mouse_move(viewer, event):
    """Callback to add points while moving the mouse if drawing is active."""
    global current_polygon
    
    if not drawing_active:
        return  # Do nothing if not active
    
    # Get cursor position in data coordinates using the event parameter
    pos = event.position
    if pos is None:
        return  # Mouse not over image
    
    # Take only the visible dimensions (x, y)
    pos = pos[:2]
    
    # Ensure only adding points if far enough from the last one
    if not current_polygon or np.linalg.norm(np.array(pos) - np.array(current_polygon[-1])) > min_distance:
        current_polygon.append(pos)
        if current_polygon:  # Make sure we have points
            polygon_layer.data = [np.array(current_polygon)]  # Update polygon

# Update your toggle_drawing function:
def toggle_drawing(viewer):
    """Toggles the drawing mode on/off when 'p' is pressed."""
    global drawing_active, current_polygon
    
    drawing_active = not drawing_active  # Toggle state
    
    if drawing_active:
        current_polygon.clear()  # This is fine if current_polygon is already initialized
        polygon_layer.data = []  # Clear the layer
        
        # Connect the mouse move callback to the viewer
        if on_mouse_move not in viewer.mouse_move_callbacks:
            viewer.mouse_move_callbacks.append(on_mouse_move)
        print("Drawing mode activated")
        
    else:
        
        # Remove the callback when not drawing
        if on_mouse_move in viewer.mouse_move_callbacks:
            viewer.mouse_move_callbacks.remove(on_mouse_move)
        print("Drawing mode deactivated")

# Now open the image in Napari
viewer = napari.Viewer()
viewer.add_image(img)
# Add an empty shapes layer for the polygon
polygon_layer = viewer.add_shapes(shape_type='polygon', edge_color='red', face_color='transparent')

# Attach event listener for the 'd' key
# Code seems to be correct -- https://napari.org/stable/gallery/custom_key_bindings.html
@viewer.bind_key('p')
def toggle_drawing_key(viewer):
    toggle_drawing(viewer)



# CONTINUE BELOW, WHY DOESN'T THIS PRINT SOMETHING???
# SEE ALSO: https://napari.org/stable/gallery/custom_key_bindings.html



@viewer.bind_key('t') # test function
def test_fn(viewer):
    print('just testing')

napari.run()