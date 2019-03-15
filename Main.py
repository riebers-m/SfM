'''
author: Maximilian Rickers
date:   11.03.2019

Discription: 3D Scene Reconstruction with Structure from Motion

TODO:

'''

import cv2
from Camera import Camera
from SfM import SceneReconstruction3D


if __name__ == "__main__":
    camera = Camera("D415")
    # initialize camera
    camera.setUp_RS(streams=["color","depth"])
    # get intrinsic camera matrix and distortion params
    K, dist = camera.get_intrinsic()
    scene = SceneReconstruction3D(K,dist)
    # init image windows
    cv2.namedWindow('Color')
    cv2.resizeWindow('Color',800,600)
    cv2.moveWindow('Color',150,150)

    cv2.namedWindow('Depth')
    cv2.resizeWindow('Depth', 800, 600)
    cv2.moveWindow('Depth', 1000, 150)

    try:
        while(True):
            # get frames
            color_image = camera["color"]
            depth_image = camera["depth"]

            ## show images
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # show color/depth image
            cv2.imshow('Color', color_image)
            cv2.imshow('Depth', depth_colormap)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            # press 'p' for capturing frame
            captured_imgs = scene.capture_img(color_image,key)
            # after two frames captured calc feature points and fundamental and essential matrix
            if captured_imgs > 1:
                scene.plot_rectified_images()
    finally:
        camera.close()