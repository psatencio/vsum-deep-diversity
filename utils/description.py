'''
    description: Utility scripts for video frame feature extraction using pretrained DCN models
    and GLoVe.
    author: Pedro Atencio Ortiz
    mail: pedroatencio@itm.edu.co
    date: 16-07-2018
    dependencies:
        - Keras 2.2.0
        - Tensorflow 1.8.0
'''

import os
import re

import numpy as np
import skvideo.io
from skimage.transform import resize
from video import vsampling_uniform

from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
from keras.preprocessing import image

from keras.applications import vgg16, vgg19
from keras.applications import xception
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import inception_resnet_v2
from keras.applications import densenet


def read_video(video_path):
    """
    Returns video frames from input video, sampled to 1 frames per second.
    input:
        path to the video.
    output:
        numpy array of video frames.
    """

    try:
        video_generator = skvideo.io.vreader(video_path)
        metadata = skvideo.io.ffprobe(video_path)
    except:
        print "Video reading error."
        return None

    irate = int(metadata['video']['@avg_frame_rate'].split('/')[0])
    print "Detected frame rate: ", irate 

    if(irate  > 100):
        print "Correcting frame rate"
        irate = int(irate / 1000)

    video_frames = vsampling_uniform(video_generator, slide_width=1, shift=1, irate=irate, frate_s=1)
    print(str(len(video_frames))+' frames extracted.') 

    return (irate, video_frames)

def load_dcn_model(model_name = 'VGG16'):
    """
    Returns a pretrained keras DCN model. For each model we return complete activation (last layer)
    and intermediate activation (penultimate layer).
    model_name can be:
        - VGG16
        - VGG19
        - xception
        - inception_resnet_v2
        - resnet50
        - inceptionV3
        - densenet
    """

    print "Loading model "+model_name

    if(model_name == 'VGG16'):
        model = vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        int_model = Model(inputs=model.input, outputs = model.get_layer('fc2').output)
    elif(model_name == 'VGG19'):
        model = vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        int_model = Model(inputs=model.input, outputs = model.get_layer('fc2').output)
    elif(model_name == 'xception'):
        model = xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        int_model = Model(inputs=model.input, outputs = model.get_layer('avg_pool').output)
    elif(model_name == 'inception_resnet_v2'):
        model = inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        int_model = Model(inputs=model.input, outputs = model.get_layer('avg_pool').output)
    elif(model_name == 'resnet50'):
        model = resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        int_model = Model(inputs=model.input, outputs = model.get_layer('flatten_1').output)
    elif(model_name == 'inceptionV3'):
        model = inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        int_model = Model(inputs=model.input, outputs = model.get_layer('avg_pool').output)
    elif(model_name == 'densenet'):
        model = densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        int_model = Model(inputs=model.input, outputs = model.get_layer('avg_pool').output)
    else:
        print "Model not found."
        return None
    
    print "Model loaded."

    return (model, int_model)
def preprocess_image(img, size=(224,224), model="VGG16"):
    '''
    This function takes an ndarray containing an image and returns a preprocessed
    version as required by classification model.

    Arguments:
        img -- numpy ndarray containing a color image.
        target_size -- tuple of values (width, height) for image resize.
        model -- str for classifier model {vgg16, vgg19}
    
    Output:
        x: preprocessed array as required properties by model.
    '''

    x = resize(img, output_shape=(224, 224))
    x = np.expand_dims(x, axis=0)
    x = x.astype(np.float64)*255.0

    if(model == "VGG16"):
        from keras.applications.vgg16 import preprocess_input
        x = preprocess_input(x)
    elif(model == "VGG19"):
        from keras.applications.vgg19 import preprocess_input
        x = preprocess_input(x)
    elif(model == "ResNet50"):
        from keras.applications.resnet50 import preprocess_input
        x = preprocess_input(x)
    elif(model == "DenseNet"):
        from keras.applications.densenet import preprocess_input
        x = preprocess_input(x)
    elif(model == "Xception"):
        from keras.applications.xception import preprocess_input
        x = preprocess_input(x)
    elif(model == "InceptionV3"):
        from keras.applications.inception_v3 import preprocess_input
        x = preprocess_input(x)
    elif(model == "InceptionResNetV2"):
        from keras.applications.inception_resnet_v2 import preprocess_input
        x = preprocess_input(x)
    else:
        print("Model not recognized.")
        return None

    return x

def get_visual_representation(model, int_model, video_frames):
    '''
    Returns DCN last and penultimate layer activation for each frame in
    input video. Last layer activation consist in a tuple of values 
    (category_id(imagenet), probability, label). Penultimate layer is 
    generally a 4096 dimensional array. 

    Arguments:
        model -- Keras model.
        int_model -- Keras model.
        video_frames -- list of video frames.
    
    Output:
        model_activations -- activations from model's last layer.
        int_model_activations -- activations from model's penultimate layer.
    '''
    model_activations = []
    int_model_activations = []

    for frame in video_frames:
        x = preprocess_image(frame)
        pred = model.predict(x)
        styled_pred = decode_predictions(pred, top=5)[0]

        int_pred = int_model.predict(x)

        model_activations.append(styled_pred)
        int_model_activations.append(int_pred)

    return (model_activations, int_model_activations)

def load_embedding_matrix(wb_path):
    '''
    Returns a pretrained word embedding matrix in path wb_path, 
    as a dictionary: word:vector
    '''
    #BASE_DIR = ''
    #GLOVE_DIR = BASE_DIR+'pretrained-glove/'
    #EMBEDDING_DIM = 100

    '''
    pretrained GLOVE model loading
    '''
    embedding_matrix = {}

    f = open(wb_path)

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_matrix[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embedding_matrix))
    
    return embedding_matrix
def create_text(labels):
    '''
    Transform a list of lists of words to a list of text with
    words separated by space char.

    This is requered for GloVe classificator to read the text composed
    by categories detected by DCN model.    
    '''
    filtered_labels = []

    for frame_labels in labels:
        key_words = []
        for label in frame_labels:
            
            #words = label[1].split("_")
            words = re.split('\W+|_', label[1]) #split words in the form w1_w2 to w1, w2
            
            if(len(words) > 1):
                for word in words:
                    #key_words.append(word.lower())
                    key_words.append( (word.lower(), label[2]) ) #list if tuples (word, probability)
            else:
                key_words.append( (label[1].lower(), label[2]) )

        filtered_labels.append(key_words)

    labels_strings = []

    for text in filtered_labels:
        
        stringText = ""
        
        for word in text:
            stringText += " "+str(word[0])
        
        labels_strings.append(stringText)

    return (labels_strings, filtered_labels)

def get_categorical_representation(embedding_matrix, model_activations, verbose=False):
    EMBEDDING_DIM = 100

    (captions, category_info) = create_text(model_activations)

    GLOVE_frames = []

    for i in range(np.shape(captions)[0]):   # Video Frames         
        GLOVE_words = np.zeros([1, EMBEDDING_DIM], dtype=np.float)
        words = captions[i].split(' ')

        c = 0
        j = 0
        
        for word in words: # Words in Frame
            if(len(word) > 0):
                try:
                    #Each word or category is multiplied by it probability to obtain
                    #a weighted average
                    prob = category_info[i][j][1]
                    #prob = 1
                    GLOVE_words = GLOVE_words + prob * embedding_matrix[word]
                    c = c + 1
                except KeyError:
                    if(verbose):
                        print("word: "+word+", not found in corpus.")
                
                j = j + 1

        GLOVE_words = GLOVE_words / float(c)
        GLOVE_frames.append(GLOVE_words)

    return GLOVE_frames

def activations_difference(activations, function='diff'):
    (m,n) = activations.shape
    result = np.zeros([m-1])
    
    if(function=='diff'):
        result = np.linalg.norm(np.abs(np.diff(norm_data(activations), axis=0)), axis=1)
        return result
    
    elif(function=='cos'):
        for i in range(m-1):
            cosine_similarity = np.dot(activations[i,:], activations[i+1,:]) / (np.linalg.norm(activations[i,:]) * np.linalg.norm(activations[i+1,:]))
            
            if(cosine_similarity > 1.0):
                cosine_similarity = 1.0
            
            result[i] = np.arccos(cosine_similarity) / np.pi
            
        return result
    
    else:
        print("function not recognized. Try with diff or cos.")
        return

def normalize_data(x):
    minx = np.min(x)
    maxx = np.max(x)
    
    return (x - minx) / (maxx - minx)
def norm_data(x):
    return x / np.linalg.norm(x)

def get_visual_diversity(int_model_activations):
    '''
    Computes visual diversity as the temporal difference between DCN penultimate layer
    activations frame by frame.
    '''
    activations = np.squeeze(int_model_activations)
    activations_diff = activations_difference(activations, function='diff')
    activations_diff_normalized = normalize_data(activations_diff)

    return activations_diff_normalized
def get_categorical_diversity(categorical_activations):
    
    GLOVE_activations_diff = activations_difference(categorical_activations, function='diff')
    GLOVE_activations_diff_normalized = normalize_data(GLOVE_activations_diff)

    return GLOVE_activations_diff_normalized
def get_combined_diversity(visual_diversity, categorical_diversity):
    return normalize_data(0.5*visual_diversity + 0.5*categorical_diversity)

'''
Example
'''

"""
#1. video loading and processing
video_path = "../video_dataset/SumMe/videos/St Maarten Landing.webm"
(irate, video_frames) = read_video(video_path)

#2. computer visual representations
model_name = "inceptionV3"
(model, int_model) = load_dcn_model(model_name)
(model_activations, int_model_activations) = get_visual_representation(model, int_model, video_frames)

#3. compute categorical representations
wb_path = '../pretrained-glove/glove.6B.100d.txt'
embedding_matrix = load_embedding_matrix(wb_path)
categorical_activations = get_categorical_representation(embedding_matrix, model_activations)
"""