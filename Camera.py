import numpy as np
import cv2
import pyrealsense2 as rs

class Camera:
    '''
    Implementation of RealSense Camera Class
    '''

    def __init__(self,name):
        '''
        RealSense Camera typ
        :param name: Typ of RS Camera (d415/d430/...)
        '''
        self.name = name

    def setUp_RS(self,width=640,height=480,fps=30,streams="color"):
        '''
        Initialization if RS Camera

        :param width: int
            Width of camera image
        :param height: int
            Height of camera image
        :param fps: int
            Frames per Second
        :param streams: str
            List if used Camera streams. Choose between "color", "ir", "depth"
        '''
        # Configure streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        if "color" in streams:
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if "ir" in streams:
            raise NotImplementedError("Try to implement ir stream config")
        if "depth" in streams:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Start streaming
        self.pipeline.start(config)

    def _camera_params(self):
        '''
        Returns stream profile and camera intrinsics for color stream
        :return: np.array
            intr: intrinsic camera matrix
            dist: distortion coefficiants
        '''

        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.intrinsics = color_profile.get_intrinsics()

        # return numpy array of intrinsic camera params
        intr = np.eye(3)
        intr[0, 0] = self.intrinsics.fx
        intr[1, 1] = self.intrinsics.fy
        intr[0, 2] = self.intrinsics.ppx
        intr[1, 2] = self.intrinsics.ppy
        intr = intr.reshape(3, 3)

        # distortion params
        dist = np.asanyarray(self.intrinsics.coeffs).reshape(1, 5)

        return intr, dist

    def __getitem__(self, item):
        '''
        Methode to return image array
        :param item: str
            Image typ
        :return: np.array
            Return array of image typ
        '''
        # get frames
        frames = self.pipeline.wait_for_frames()

        if item == "color":
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            return color_image
        elif item == "depth":
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            return depth_image

    def close(self):
        self.pipeline.stop()

    def get_intrinsic(self):
        return self._camera_params()


