import os
import random
import json
from pytorch_svgrender.painter.dream3dvg.scene.dataset_readers import GenerateRandomCameras,GeneratePurnCameras, GenerateCircleCameras, readCircleCamInfo
from pytorch_svgrender.utils.camera_utils import camera_to_JSON, cameraList_from_RcamInfos, loadRandomCam

class Scene:
    
    def __init__(self, args, save_path=None, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.args = args
        self.pose_args = args.x.camera_param
        self.model_path = save_path
        self.loaded_iter = None
        self.resolution_scales = resolution_scales
        
        self.test_cameras = {}
        
        scene_info = readCircleCamInfo(self.model_path, self.pose_args)
        
        self.init_scene_info = scene_info
        
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = self.pose_args.default_radius #    scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales:
            self.test_cameras[resolution_scale] = cameraList_from_RcamInfos(scene_info.test_cameras, resolution_scale, self.pose_args)

    def getRandTrainCameras(self, batch=10, scale=1.0):
        rand_train_cameras = GenerateRandomCameras(self.pose_args, batch, SSAA=True)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args, SSAA=True)   
             
        return train_cameras[scale]


    def getPurnTrainCameras(self, scale=1.0):
        rand_train_cameras = GeneratePurnCameras(self.pose_args)
        train_cameras = {}
        for resolution_scale in self.resolution_scales:
            train_cameras[resolution_scale] = cameraList_from_RcamInfos(rand_train_cameras, resolution_scale, self.pose_args)        
        return train_cameras[scale]


    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getCircleVideoCameras(self, scale=1.0, batch_size=120, render45 = True):
        video_circle_cameras = GenerateCircleCameras(self.pose_args,batch_size,render45) # len(600) * {R,T,fovx,fovy,w,h,delta_polar,delta_azimuth,delta_adius}
        video_cameras = {}
        for resolution_scale in self.resolution_scales:
            video_cameras[resolution_scale] = cameraList_from_RcamInfos(video_circle_cameras, resolution_scale, self.pose_args)        
        return video_cameras[scale] # len(600)
    
    def getSingleCamera(self, cam_info):
        camera = loadRandomCam(opt=self.pose_args, id=cam_info.uid, cam_info=cam_info, resolution_scale=self.resolution_scales[0])
        return camera