
            

def edgedetectionstuff():

    # some edge detection stuff
    # trying out something else
    tiles_maxfiltsmall = [maximum_filter(tile, size=20) for tile in tiles]
    tiles_norm_small = [tile/tile_mf for tile,tile_mf in zip(tiles, tiles_maxfiltsmall)]
    plt.imshow(tiles_norm_small[0]); plt.show(); plt.close()

    # apply sobel for edge detection on tile 0
    sobel_tile = sk.filters.sobel(tiles[0])
    plt.imshow(sobel_tile); plt.show(); plt.close()

    # resize tile 0 to 20x smaller using max
    tile_0_small = sk.transform.rescale(tiles[0], 0.2, order=0, anti_aliasing=False)
    # apply sobel
    sobel_tile_0_small = sk.filters.sobel(tile_0_small)
    plt.imshow(sobel_tile_0_small); plt.show(); plt.close()

        
    # now identify the tile with the most variance
    # for each of the tiles, apply a local max filter of size 250




def annotate_pictures_REMOVE(input_folder, output_folder=None):
    # input_folder = '/Users/m.wehrens/Data_UVA/2024_07_Wang-cel/2025_Cells_preliminarybatch1/Cheeck-Cells_AnnotatedMW_resized/'
    ''''
    Loop over the pictures in the folder, open them with Napari,
    create an annotation layer, allow user to create polygons,
    save the layer to a numpy file with similar filename
    and _seg.
    '''
    
    if output_folder==None:
        output_folder = input_folder.rstrip('/') + '_humanannotated/'
    os.makedirs(output_folder, exist_ok=True)
    
    # again, loop over the pictures in the folder
    list_all_files = glob.glob(input_folder + '/*')
    
    for idx, filepath in enumerate(list_all_files):
        # idx=0; filepath=list_all_files[0]
        filename = filepath.split('/')[-1]
        
        img = Image.open(filepath)
        img = np.array(img)
        
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # invert image scale
        img_greyscale = 255 - img_greyscale

        # determine otsu threshold on img_greyscale
        thresholdval        = threshold_triangle(img_greyscale)
        img_segmaskauto     = img_greyscale > thresholdval
        # img_segmask_o  = binary_opening(img_segmask_o, disk(10))
        # use cv2 to perform binary opening
        img_segmaskauto = cv2.morphologyEx(img_segmaskauto.astype(np.uint8), cv2.MORPH_CLOSE, disk(20))
        # apply morphological opening from skimage library
        
        
        # plt.imshow(img_segmask_o); plt.show(); plt.close()
        
        viewer = napari.Viewer()
        viewer.add_image(img_greyscale)
        # add a label layer
        #seg_layer = viewer.add_labels(name='segmentation', data=np.zeros(np.shape(img)[:2], dtype=np.uint8))
        seg_layer = viewer.add_labels(name='segmentationauto', data=img_segmaskauto)

        # viewer.close()
        
        
        seg_mask = viewer.layers['segmentation'].data

        # let's threshold at 1% of the cell values
        threshold    = np.percentile(img_greyscale[seg_mask==1], 0.98)
        img_cellmask = np.zeros_like(img_greyscale)
        img_cellmask[img_greyscale<threshold] = 1
        

        
        
        
        plt.hist(img_greyscale[seg_mask==1].flatten(), bins=100, label='background')
        plt.hist(img_greyscale[seg_mask==2].flatten(), bins=100, label='foreground')
        plt.axvline(threshold)
        plt.legend()
        plt.show(); plt.close()


        
        viewer = napari.Viewer()
        viewer.add_image(img)
        # add a label layer
        seg_layer = viewer.add_labels(name='mask_thresholded', data=img_cellmask)
        

        seg_layer.data
        viewer.layers['empty_labels'].data

        # retrieve the labeled layer
        plt.imshow(img_cellmask); plt.show(); plt.close()
        
        napari.run()
        
        # close napari
        viewer.close()
        
        # save the polygon layer to a numpy file
        np.save(output_folder + filename + '_seg.npy', polygon_layer.data)
        
        # create an image of equal size as img
        img_seg = np.zeros_like(img)
        # now draw the polygons from the polygon_layer on top
        for polygon in polygon_layer.data:
            # polygon is a list of points
            # we need to convert it to a list of tuples
            polygon = [(int(point[0]), int(point[1])) for point in polygon]
            # now draw the polygon on img_seg
            img_seg = cv2.fillPoly(img_seg, [np.array(polygon)], 1)