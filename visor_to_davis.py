import os
import json
import argparse
from utils.vis import *
import shutil
import glob
import json
from tqdm import tqdm


global_keys={} # store the keys of all the videos
sequences = set() # store the set of sequences 

#the unseen kitchens in train
unseen_kitchens = ['P07_101','P07_103','P07_110','P09_02','P09_07','P09_104','P09_103','P09_106','P21_01','P21_01','P29_04']


def json_to_masks(filename,output_directory,images_root,object_keys=None,output_resolution="854x480"):
    """
    json_to_masks store the images and generate masks of a given video json
    :param filename: path to the json file
    :param output_directory: path to save the output data
    :param images_root: path to VISOR images
    :param object_keys: dict of each sequence with the it's set of objects and their color codes
    :param output_resolution: output resolution of the masks and images
    :return: None
    """ 

    #get the output images dimensions
    height = int(output_resolution.split('x')[1])
    width = int(output_resolution.split('x')[0])
    #create folder to save the data
    os.makedirs(output_directory, exist_ok=True)
    global sequences
    #get the annotations
    f = open(filename)
    data = json.load(f)
    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data['video_annotations'], key=lambda k: k['image']['image_path'])

    full_path=""
    for datapoint in data:
        image_name = datapoint["image"]["name"]
        image_path = datapoint["image"]["image_path"]
        seq_name = datapoint["image"]["subsequence"]
        masks_info = datapoint["annotations"]
        full_path =output_directory+'/' +seq_name+'/'#until the end of sequence name
        #create the folders for the sequences to store images and masks
        os.makedirs(full_path,exist_ok= True)
        os.makedirs(full_path.replace('Annotations','JPEGImages'), exist_ok=True,mode=0o777)
        #scale the images corresponding to the output resolution
        img1 = cv2.imread(os.path.join(images_root,datapoint["image"]["video"]+'/'+image_name))
        resized1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(full_path.replace('Annotations','JPEGImages'),image_name),resized1)
        #generate the masks and get the key values
        object_keys_values = generate_masks(image_name, masks_info, full_path,object_keys[seq_name],(width,height)) 
        global_keys[seq_name] = object_keys_values
        sequences.add(full_path[:-1])
def folder_of_jsons_to_masks(input_directory,output_directory,images_root,mapping_file,k,keep_first_frame_masks_only=False,output_resolution="854x480"):
    """
    folder_of_jsons_to_masks it go through set of json objects and generate their corresponding PNG masks, it also saves the color mapping of the 'object to code' so you can build the jsons back from the PNGs 
    :param input_directory: path to the json files
    :param output_directory: path to save the output data
    :param images_root: path to VISOR images
    :param mapping_file: path to save the mapping file
    :param k: min number of files per sequence in order to keep the sequnece, otherwise it would be deleted
    :param keep_first_frame_masks_only: wheather to filter out the masks that are not in the first frame of each sequence
    :param output_resolution: output resolution of the masks and images
    :return: None
    """ 
    
    #go through all videos json files
    for json_file in tqdm(sorted(glob.glob(input_directory + '/*.json'))):
        
        if keep_first_frame_masks_only:
            #get set of objects of the first frame of each sequence
            objects = get_first_frame_objects (json_file)
            
        else:
            #get set of objects of each sequence
            objects = get_sequence_objects(json_file)
        
        #generate the masks and images of the json file
        json_to_masks(json_file,output_directory,images_root,objects,output_resolution)

    #path to save the txt file containing the considered sequences
    file_of_seq = os.path.join('/'.join(output_directory.split('/')[:-2]),'ImageSets/2022/'+os.path.basename(input_directory)+'.txt')
    
    #if it's val set, store the unseen sequences, they would be used to evaluate the unseen kitchens
    if os.path.basename(input_directory) == 'val':
        
        #store sequences with k images of more and return back the considered unseen sequences
        unseen_sequences = filter_sequences_with_less_than_k(sequences,file_of_seq,k,include_unseen=True)
        #path to save the txt file containing the considered unseen sequences
        file_of_seq = os.path.join('/'.join(output_directory.split('/')[:-2]),'ImageSets/2022/'+os.path.basename(input_directory)+'_unseen.txt')
        #store the unseen sequences
        textfile = open(file_of_seq, "w")
        for element in sorted(unseen_sequences):
            textfile.write(element)
            if unseen_sequences.index(element) != (len(unseen_sequences)-1):
               textfile.write('\n') 
        textfile.close()       
    else:
        #store sequences with k images of more
        filter_sequences_with_less_than_k(sequences,file_of_seq,k,include_unseen=False)

    #store the color mapping of the data, it would be needed if you want to convert the PNGs back to JSON for codalab submission 
    out_file = open(mapping_file, "w")
    json.dump(global_keys, out_file)
    out_file.close()

def filter_sequences_with_less_than_k(sequences,file_of_seq, k,include_unseen=False):
    """
    filter_sequences_with_less_than_k it cleans the data as it would consider the sequences with more or equal to k files only. consdierd sequences would be written in a txt file, also it could return the unseen sequences if include_unseen param is set
    :param sequences: set of all sequences in the video
    :param file_of_seq: a txt file where the sequences would be stored
    :param include_unseen: is to return back the sequences of the unseen kitchens
    :return: the sequences of the unseen kitchens if include_unseen=True, otherwise []
    """ 
    global unseen_kitchens
    unseen_sequences = []
    print('Data cleaning . . . ')
    os.makedirs('/'.join(file_of_seq.split('/')[:-1]),exist_ok= True)
    #get stats of the sequences and consider the ones with more or equal k files
    files,included_sequences = find_number_of_images_per_seq(sequences,k)
    print(f'Number of sequences with less than {k} images is {len(files)} (deleted)')
    print(f'Number of sequences AFTER cleaning is {len(included_sequences)}')

    #write the considered(included) sequences into a txt file
    textfile = open(file_of_seq, "w")
    included_sequences = sorted(included_sequences) 
    for element in sorted(included_sequences):
        textfile.write(element)
        #if you want to store a list of unseen kitchens' sequences 
        if include_unseen:
            if '_'.join(element.split('_')[:2]) in unseen_kitchens: # get the video ID and check if it's part of the unseen kitchens (predefined)
                unseen_sequences.append(element)
        
        #if it's not the last element, add new line (to avoid adding blank like at the end)
        if included_sequences.index(element) != (len(included_sequences)-1):
           textfile.write('\n') 
    textfile.close()
    return unseen_sequences # return it if asked in include_unseen param

def find_number_of_images_per_seq(sequences,k):
    """
    find_number_of_images_per_seq calculate number of frames in each sequence, just consider the ones with at least k frames and remove the images and masks otherwise
    :param file: sequences of sequences to check
    :param k: minimum number of files in a sequence to be considered 
    :return: a dictionary of each sequence with less than k frames with it's number of removed frames as a value. also return the list of considered sequences as a list
    """ 
    files = []
    included_sequences = []
    for seq in sequences:
        num_files = len(glob.glob(seq+'/*.png'))
        #if the number of frames in the sequence less than k, then remove all it's images and masks as it would not be considered
        if num_files < k:
            #store the number of removed files in each sequence
            files.append({seq.split('/')[-1]:num_files})
            #remove the masks
            if os.path.exists(seq):
                shutil.rmtree(seq)
            #remove the images
            if os.path.exists(seq.replace('Annotations','JPEGImages')):
                shutil.rmtree(seq.replace('Annotations','JPEGImages'))

        else:
            # the sequence would be considerd as part of the dataset
            included_sequences.append(seq.split('/')[-1])
            
    return files,included_sequences 

def get_sequence_objects(file):
    """
    get_sequence_objects gets set of objects of each seqeunce. This would help to get the color codes for each object (objects sorted alphabetically)
    :param file: is the json file which would be part of VISOR dataset
    :return: a dictionary of sequence name as a key and set of sorted objects with their color codes as value
    """ 
    objects=set()
    f = open(file)
    # returns JSON object as a dictionary
    data = json.load(f)

    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data['video_annotations'], key=lambda k: k['image']['image_path'])

    # Iterating through the json list
    prev_seq = "" # this will help to catch the next seqeunce
    masks_per_seq = {} # this would return the objects per sequence
    for datapoint in data:
        seq = datapoint['image']['subsequence']
        # if there's a new sequence, then get the objects of that seqeunce
        if (seq != prev_seq):
            if prev_seq != "":
                objs_elements = sorted(objects)
                key = 1
                object_maps = {}
                for objs_element in objs_elements:
                    object_maps[key] = objs_element
                    key += 1
                masks_per_seq[seq] = object_maps
                objects = set()
            prev_seq = seq

        #get the objects in the frame, it would be appedned to the set of remaining objects in the sequence
        masks_info = datapoint["annotations"]
        entities = masks_info
        for entity in entities: #loop over each object
            object_annotations = entity["segments"]
            if not len(object_annotations) == 0: #if there is annotation for this object, add it
                objects.add(entity["name"])

        #if there is still objcts (to include the last sequence of the file as it has not covered in the last loop)
        if len(objects) != 0:
            objs_elements = sorted(objects)
            key = 1 # the color maps starts from 1 and ends with the number of objects
            object_maps = {}
            for objs_element in objs_elements:
                object_maps[key] = objs_element
                key += 1
            masks_per_seq[seq] = object_maps

    return masks_per_seq

def get_first_frame_objects(file):
    """
    get_first_frame_objects gets set of objects in the first frame of each seqeunce. This would help to get the color codes for each object (objects sorted alphabetically)
    :param file: is the json file which would be part of VISOR dataset
    :return: a dictionary of sequence name as a key and set of sorted objects with their color codes as value
    """ 

    objects=set() #set of objects of the first frame
    f = open(file)
    data = json.load(f)
    
    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data['video_annotations'], key=lambda k: k['image']['image_path'])
    prev_seq = "" # this will help to catch the next seqeunce
    masks_per_seq = {} # this would return the objects per sequence

    for datapoint in data:
        seq = datapoint['image']['subsequence']
        
        # if there's a new sequence, then get the objects of that seqeunce
        if (seq != prev_seq): 
            prev_seq = seq
            masks_info = datapoint["annotations"]
            entities = masks_info
            for entity in entities: #loop over each object
                object_annotations = entity["segments"]
                if not len(object_annotations) == 0: #if there is annotation for this object, add it
                    objects.add(entity["name"])
            
            #sort them to get the same color maps in each run
            objs_elements = sorted(objects)
            objects = set()
            key = 1 # the color maps starts from 1 and ends with the number of objects
            object_maps = {}
            for objs_element in objs_elements:
                object_maps[key] = objs_element
                key += 1
            masks_per_seq[seq] = object_maps

    return masks_per_seq

if __name__ == "__main__":
    def get_arguments():
        parser = argparse.ArgumentParser(description="parameters for VISOR to DAVIS conversion")
        parser.add_argument("-set", type=str, help="train, val", required=True)
        parser.add_argument("-keep_first_frame_masks_only", type=int, help="this flag to keep all masks or the masks in the first frame only, this flag usually 1 when generating VAL and 0 when generating Train", required=True)
        parser.add_argument("-visor_jsons_root", type=str, help="path to the json files of visor",default='../VISOR')
        parser.add_argument("-images_root", type=str, help="path to the images root directory",default='../VISOR_images')
        parser.add_argument("-output_directory", type=str, help="path to the directory where you want VISOR to be",default='../data')
        parser.add_argument("-output_resolution", type=str, help="resolution of the output images and masks",default='854x480')

        return parser.parse_args()

    args = get_arguments()

    visor_set = args.set
    visor_jsons_root = args.visor_jsons_root
    output_directory = args.output_directory
    images_root = args.images_root
    keep_first_frame_masks_only = False if args.keep_first_frame_masks_only == 0 else True
    output_resolution = args.output_resolution
    height = output_resolution.split('x')[1]+'p' # resolution of the output (correspoding to the height) - it would be used to save it in relevent folder

    #path where the mappping between the generated mask color codes and the corrspoding object names would be stored
    mapping_file = os.path.join(os.path.join(output_directory,'VISOR_2022'),visor_set+'_data_mapping.json') 

    if os.path.exists(mapping_file):
        os.remove(mapping_file)


    print('Converting VISOR to DAVIS . . .')
    if visor_set =='val':
        if not keep_first_frame_masks_only:
            print('Warning!!, usually "keep_first_frame_masks_only" flag is True when generating Val except if you want to generate the data to train on Train/val')
        folder_of_jsons_to_masks(os.path.join(visor_jsons_root,visor_set), os.path.join(output_directory,'VISOR_2022/Annotations/'+height),images_root,mapping_file,2,keep_first_frame_masks_only,output_resolution)

    elif visor_set =='train':
        if keep_first_frame_masks_only:
            print('The "keep_first_frame_masks_only" flag should be False when generating Train!! please double check!!')
        folder_of_jsons_to_masks(os.path.join(visor_jsons_root,visor_set), os.path.join(output_directory,'VISOR_2022/Annotations/'+height),images_root,mapping_file,3,keep_first_frame_masks_only,output_resolution)
