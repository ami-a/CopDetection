"""loading the detection model variables for the detector object"""
import tensorflow as tf
from TrackEverything.tool_box import DetectionVars

def get_model(det_model_path):
    """Get the model obj

    Args:
        det_model_path (tf.model): path to model

    Returns:
        [type]: [description]
    """
    print("loading detection model...")
    det_model=tf.saved_model.load(det_model_path)
    det_model = det_model.signatures['serving_default']
    print("detection model loaded!")
    return det_model

#custome detection model interpolation
DETECTION_THRESHOLD=0.5
def custome_get_detection_array(
        detection_var,
        image,
        detection_threshold=DETECTION_THRESHOLD,
    ):
    """A method that utilize the detection model and return an np array of detections.
    Each detection in the format [confidence,(xmin,ymin,width,height)]
    Args:
        detection_var (DetectionVars): the DetectionVars obj
        image (np.ndarray): current image
        detection_threshold (float): detection threshold
    """
    detection_model=detection_var.detection_model
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

#detection variables
def get_det_vars(det_model_path):
    """loading the detection model variables for the detector object
    We define here the model interpolation function so the detector
    can use the model

    Args:
        det_model_path (str): The model path

    Returns:
        DetectionVars: the detection model variables
    """
    return DetectionVars(
        detection_model=get_model(det_model_path),#custome loading the detection model
        #and only providing the model to the DetectionVars
        detection_proccessing=custome_get_detection_array,
        detection_threshold=DETECTION_THRESHOLD
    )
