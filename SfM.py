import numpy as np
import cv2
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class SceneReconstruction3D:
    '''
    Class for 3D reconstruction with Structure from Motion
    '''
    def __init__(self,K,dist):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.dist = dist
        self.imgs = []
        self.mode = "ORB"

    def capture_img(self,img, key):
        '''
        Capture two images for 3D reconstruction

        :param img: np.array
            Captured image
        :param key: int
            Representation of character in Unicode
        :return:
        '''
        assert len(img) > 3, "Please capture RGB image"
        if key & 0xFF == ord('p'):
            if len(self.imgs) < 2:
                self.imgs.append(img)
                print("Captured {}/2 images".format(len(self.imgs)))
            else:
                print("Already captured " + str(len(self.imgs)) + " images")
                print("Please press 'c' for reset")
        elif key & 0xFF == ord('c'):
            self.imgs = []
            print("Please capture two images")

        return len(self.imgs)


    def _undistore_imgs(self):
        self.imgs[0] = cv2.undistort(self.imgs[0],self.K,self.dist)
        self.imgs[1] = cv2.undistort(self.imgs[1],self.K,self.dist)


    def set_detector(self,name):
        '''
        Sets feature Detector
        :param name: str
            Name of feature Detector (only ORB is supported)
        '''
        assert type(name) != type(str), "Please choose valid feature extraction"

        self.mode = name
        print(self.mode)

    def extract_keypoints(self):
        # assert self.mode == None, "Please set extraction mode"
        if self.mode == "ORB":
            self._extract_keypoints_orb()
        elif self.mode == "flow":
            pass

    def _extract_keypoints_orb(self):
        # self._undistore_imgs()
        detector = cv2.ORB_create(nfeatures=200)
        first_key_points, first_desc = detector.detectAndCompute(cv2.cvtColor(self.imgs[0],cv2.COLOR_BGR2GRAY), None)
        second_key_points, second_desc = detector.detectAndCompute(cv2.cvtColor(self.imgs[1],cv2.COLOR_BGR2GRAY), None)
        # Brute Force matcher
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(first_desc, second_desc)

        first_match_points = np.zeros((len(matches), 2),
                                      dtype=np.float32)
        second_match_points = np.zeros_like(first_match_points)
        for i in range(len(matches)):
            first_match_points[i] = first_key_points[matches[i].queryIdx].pt
            second_match_points[i] = second_key_points[matches[i].trainIdx].pt
        self.match_pts1 = first_match_points
        self.match_pts2 = second_match_points

        # draw matches
        img3 = cv2.drawMatches(self.imgs[0], first_key_points,
                               self.imgs[1], second_key_points,
                               matches, None, flags=2)

        cv2.imshow('matches', img3)

    def _find_fundamental_matrix(self):
        self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC,
                                                    0.1,
                                                    0.99)

    def _find_essential_matrix(self):
        self.E = self.K.T.dot(self.F).dot(self.K)

    def _find_camera_matrices(self):
        U, S, Vt = np.linalg.svd(self.E)

        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

        first_inliers = []
        second_inliers = []
        for i in range(len(self.Fmask)):
            if self.Fmask[i]:
                first_inliers.append(self.K_inv.dot([self.match_pts1[i][0], self.match_pts1[i][1],1.0]))
                second_inliers.append(self.K_inv.dot([self.match_pts2[i][0], self.match_pts2[i][1],1.0]))

        # First choice: R = U * Wt * Vt, T = +u_3(See
        # Hartley & Zisserman 9.19)
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not self._in_front_of_both_cameras(first_inliers,second_inliers, R, T):
            # Second choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]
        if not self._in_front_of_both_cameras(first_inliers,second_inliers, R, T):
            # Third choice: R = U * Wt * Vt, T = u_3
            R = U.dot(W.T).dot(Vt)
            T = U[:, 2]
        if not self._in_front_of_both_cameras(first_inliers,second_inliers, R, T):
            # Fourth choice: R = U * Wt * Vt, T = -u_3
            T = - U[:, 2]

        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))
        print("Camera Matrix1 : {} \nCamera Matrix2 : {}".format(self.Rt1,self.Rt2))

    def _in_front_of_both_cameras(self, first_points, second_points, rot, trans):
        rot_inv = rot

        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :], trans) / np.dot(rot[0, :] - second[0] * rot[2, :], second)
            first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False
            return True

    def plot_rectified_images(self):
        self._extract_keypoints_orb()

        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices()

        R = self.Rt2[:, :3]
        T = self.Rt2[:, 3]

        # Image rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K,self.dist, self.K, self.dist, self.imgs[0].shape[:2], R, T,alpha=1.0)
        mapx1, mapy1 = cv2.initUndistortRectifyMap(self.K, self.dist, R1, self.K, self.imgs[0].shape[:2], cv2.CV_32F)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(self.K, self.dist, R2,self.K, self.imgs[1].shape[:2], cv2.CV_32F)
        img_rect1 = cv2.remap(self.imgs[0], mapx1, mapy1,cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(self.imgs[1], mapx2, mapy2,cv2.INTER_LINEAR)

        # Plot rectified images
        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2
        # draw horizontal lines
        for i in range(20, img.shape[0], 25):
            cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
        cv2.imshow('imgRectified', img)

        self.imgs = []