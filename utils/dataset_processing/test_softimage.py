import glob
import os
import numpy as np
from imageio import imsave
import argparse
# from utils.dataset_processing.image import DepthImage
from utils.dataset_processing import grasp, image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth images from Cornell PCD files.')
    parser.add_argument('path', type=str, help='Path to Cornell Grasping Dataset')
    args = parser.parse_args()

    di = image.Image.from_file("/home/trannguyenle/RemoteWorkingStation/ros_workspaces/ggcnn/SoftDataset/01/pear.png")
    di.show()
    print(di)
    # imsave(of_name, di.img.astype(np.float32))