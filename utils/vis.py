'''
This file is to visualise VISOR data from the provided JSON files
'''
import cv2
import numpy as np
from PIL import Image
from numpy import asarray
import os

def generate_masks(image_name, masks_info, output_directory,objects=None, output_resolution=(854,480)):
    """
    generate_masks generates the masks. The masks are being generated based on the order of the objects param
    :param image_name: name of the input image
    :param masks_info: the polygons and information of the annotations of the image
    :param output_directory: path to save the mask
    :param objects: set of objects to generate their masks in-order
    :param output_resolution: output resolution of the masks and images
    :return: dictionary that contains the color codes for each object in each sequence
    """ 

    #to store the keys of the objects (color codes)
    object_keys= {}
    #this for the output map to be saved in a json file
    object_keys_out = {}
    
    #generate the objects codes
    for key,value in objects.items():
        object_keys[value] = key
        object_keys_out[key] = value

    #empty mask, the size is the same as visor images which are full HD
    mask = np.zeros([1080,1920],dtype=np.uint8)

    entities = masks_info
    i = 1
    #loop over the objects of that frame
    for entity in entities:
        object_annotations = entity
        polygons = []
        #store the polygons
        for object_annotation in object_annotations['segments']:
            polygons.append(object_annotation)
        ps = []
        #store the polygons in one list. One object may has more than 1 polygon
        for poly in polygons:
            if poly == []:
                poly = [[0.0, 0.0]]
            ps.append(np.array(poly, dtype=np.int32))
        if object_keys:
            #set a color code for the object based on it's name (provided to the funtion in-order)
            if (entity['name'] in object_keys.keys()):
                cv2.fillPoly(mask, ps, (object_keys[entity['name']], object_keys[entity['name']], object_keys[entity['name']]))
        else:
            cv2.fillPoly(mask, ps, (i, i, i))
        i += 1

    #make sure that the mask is not empty
    if (not np.all(mask == 0)):
        image_name = image_name.replace("jpg", "png")
        data = asarray(mask)
        #scale the mask to the required output resolution
        scaled_mask = cv2.resize(data, (output_resolution[0],
                                    output_resolution[1]),
                                interpolation=cv2.INTER_NEAREST)

        scaled_mask = (np.array(scaled_mask)).astype('uint8')
        #store the image using davis color code
        imwrite_indexed(os.path.join(output_directory,image_name), scaled_mask)
    else: # delete the corresponding jpg image
        jpg_image_path = os.path.join(output_directory.replace('Annotations','JPEGImages'),image_name)
        os.remove(jpg_image_path)
    return object_keys_out


def imwrite_indexed(filename, im):
    """
    imwrite_indexed generates the masks based on david color codes
    :param filename: path to save the mask
    :param im: mask values as color codes (integers)
    :return: None
    """ 
    davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
    davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                             [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                             [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                             [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                             [0, 64, 128], [128, 64, 128]]
    color_palette = davis_palette
    assert len(im.shape) < 4 or im.shape[0] == 1  # requires batch size 1
    im = Image.fromarray(im, 'P')
    im.putpalette(color_palette.ravel())
    im.save(filename)
