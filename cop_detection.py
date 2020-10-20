"""An example for using the TackEverything package
This example uses an Object Detection model from the TensorFlow git
for detecting humans. Using a simple citizen/cop classification model
I've created using TF, it can now easly detect and track cops in a video using
a few lines of code.
The use of the TrackEverything package make the models much more accurate
and robust, using tracking features and statistics.
"""
import os
#hide some tf loading data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position
from TrackEverything.detector import Detector
from TrackEverything.tool_box import InspectorVars
from TrackEverything.statistical_methods import StatisticalCalculator, StatMethods
from TrackEverything.visualization_utils import VisualizationVars

from detection_vars import get_det_vars
from classification_vars import get_class_vars
from play_video import run_video
# pylint: enable=wrong-import-position

DET_MODEL_PATH="detection_models/faster_rcnn_inception_v2_coco_2018_01_28/saved_model"
CLASS_MODEL_PATH="classification_models/" \
"0.93855_L_0.3909_opt_RMSprop_loss_b_crossentropy_lr_0.0005_baches_20_shape_[165, 90]_auc.hdf5"

#set the detector
detector_1=Detector(
    det_vars=get_det_vars(DET_MODEL_PATH),
    class_vars=get_class_vars(CLASS_MODEL_PATH),
    inspector_vars=InspectorVars(
        stat_calc=StatisticalCalculator(method=StatMethods.EMA)
    ),
    visualization_vars=VisualizationVars(
        labels=["Citizen","Cop"],
        colors=["Green","Red","Cyan"],
        show_trackers=True,
        uncertainty_threshold=0.5,
        uncertainty_label="Getting Info"
    )
)

#Test it on a video
VIDEO_PATH="video/032.mp4"
run_video(VIDEO_PATH,(480,270),detector_1)
# from play_video import save_video
# save_video(VIDEO_PATH,(480,270),detector_1,"video/cop_032.avi")
