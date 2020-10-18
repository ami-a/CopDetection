"""An example for using the TackEverything package
This example uses an Object Detection model from the TensorFlow git
for detecting humans. Using a simple citizen/cop classification model
I've created using TF, it can now easly detect and track cops in a video using
a few lines of code.
The use of the TrackEverything package make the models much more accurate
and robust, using tracking features and statistics.
"""
import os
import numpy as np
import cv2
#hide some tf loading data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position
import tensorflow as tf
from TrackEverything.detector import Detector
from TrackEverything.tool_box import DetectionVars,ClassificationVars,InspectorVars
from TrackEverything.statistical_methods import StatisticalCalculator, StatMethods
from TrackEverything.visualization_utils import VisualizationVars

from play_video import run_video
# pylint: enable=wrong-import-position

#custome loading the detection model and only providing the model to the DetectionVars
print("loading detection model...")
DET_MODEL_PATH="detection_models/faster_rcnn_inception_v2_coco_2018_01_28/saved_model"
det_model=tf.saved_model.load(DET_MODEL_PATH)
det_model = det_model.signatures['serving_default']
print("detection model loaded!")

#custome detection model interpolation
DETECTION_THRESHOLD=0.5
def custome_get_detection_array(
        image,
        detection_model=det_model,
        detection_threshold=DETECTION_THRESHOLD,
    ):
    """A method that utilize the detection model and return an np array of detections.
    Each detection in the format [confidence,(xmin,ymin,width,height)]
    Args:
        image (np.ndarray): current image
        detection_threshold (float): detection threshold
        model (tensorflow model obj): classification model
    """
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    detections=detection_model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in detections.items()}
    output_dict['num_detections'] = num_detections
    #convert cordinates format to (xmin,ymin,width,height)
    output_dict['new_cordinates']=list(
        map(lambda x:get_box_cordinates(x,image.shape),output_dict['detection_boxes'])
    )
    #build the detection_array
    output= [
        [output_dict['detection_scores'][i],output_dict['new_cordinates'][i]]
            for i in range(num_detections) if
            output_dict['detection_scores'][i]>detection_threshold #filter low detectin score
            and output_dict['detection_classes'][i]==1 #detect only humans
            and output_dict['new_cordinates'][i][2]>0 #make sure width>0
            and output_dict['new_cordinates'][i][3]>0 #make sure height>0
    ]
    #print(output)
    return output

def get_box_cordinates(box,img_shape):
    """#convert model cordinates format to (xmin,ymin,width,height)

    Args:
        box ((xmin,xmax,ymin,ymax)): the cordinates are relative [0,1]
        img_shape ((height,width,channels)): the frame size

    Returns:
        (xmin,ymin,width,height): (xmin,ymin,width,height): converted cordinates
    """
    height,width, = img_shape[:2]
    xmin=max(int(box[1]*width),0)
    ymin=max(0,int(box[0]*height))
    xmax=min(int(box[3]*width),width-1)
    ymax=min(int(box[2]*height),height-1)
    return (
        xmin,#xmin
        ymin,#ymin
        xmax-xmin,#box width
        ymax-ymin#box height
    )

#providing only the classification model path for ClassificationVars since the default loding method
#tf.keras.models.load_model(path) will work
CLASS_MODEL_PATH="classification_models/" \
"0.93855_L_0.3909_opt_RMSprop_loss_b_crossentropy_lr_0.0005_baches_20_shape_[165, 90]_auc.hdf5"
#custome classification model interpolation
def custome_classify_detection(model,det_images,size=(90,165)):
    """Classify a batch of images

    Args:
        model (tensorflow model): classification model
        det_images (np.array): batch of images in numpy array to classify
        size (tuple, optional): size to resize to, 1-D int32 Tensor of 2 elements:
            new_height, new_width (if None then no resizing).
            (In custome function you can use model.inputs[0].shape.as_list()
            and set size to default)
    Returns:
        Numpy NxM vector where N num of images, M num of classes and filled with scores.

        For example two images (car,plan) with three possible classes (car,plan,lion)
        that are identify currectly with 90% in the currect category and the rest is devided equally
        will return [[0.9,0.05,0.05],[0.05,0.9,0.05]].
    """
    #resize bounding box capture to fit classification model
    if size is not None:
        det_images=np.asarray(
            list(
                map(
                    lambda img: cv2.resize(img, size, interpolation = cv2.INTER_LINEAR),
                    det_images
                )
            )
        )
    # for i,imgs in enumerate(det_images):
    #     cv2.imshow(f"crop{i}",imgs)
    predictions=model.predict(det_images/255.)
    #print(f"dections:{len(det_images)},pred:{predictions}")
    #if class is binary make sure size is 2
    if len(predictions)>0:
        reshaped_pred=np.ones((len(predictions),2))
        #size of classification list is 1 so turn it to 2
        for ind,pred in enumerate(predictions):
            reshaped_pred[ind,:]=pred,1-pred
        #print(reshaped_pred)
        predictions=reshaped_pred
    return predictions

#set the detector
detector_1=Detector(
    det_vars=DetectionVars(
        detection_model=det_model,
        detection_proccessing=custome_get_detection_array,
        detection_threshold=DETECTION_THRESHOLD
    ),
    class_vars=ClassificationVars(
        class_model_path=CLASS_MODEL_PATH,
        class_proccessing=custome_classify_detection
    ),
    inspector_vars=InspectorVars(
        stat_calc=StatisticalCalculator(method=StatMethods.EMA)
    ),
    visualization_vars=VisualizationVars(
        labels=["Citizen","Cop"],
        show_trackers=True,
        uncertainty_threshold=0.5,
        uncertainty_label="Getting Info"
    )
)

#Test it on a video
VIDEO_PATH="video/024.mp4"
run_video(VIDEO_PATH,(480,270),detector_1)
