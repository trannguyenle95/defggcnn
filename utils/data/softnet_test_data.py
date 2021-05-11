import os
import glob

from .test_data import TestDatasetBase
from utils.dataset_processing import grasp, image


class SoftNetTestDataset(TestDatasetBase):
    """
    Dataset wrapper for the SoftNet test dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: SoftNet Test Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(SoftNetTestDataset, self).__init__(**kwargs)

        depthf = glob.glob(os.path.join(file_path, '*', '*d.png'))
        depthf.sort()
        l = len(depthf)

        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            depthf = depthf[int(l*ds_rotate):] + depthf[:int(l*ds_rotate)]
        
        stiffnessf = [f.replace('d.png', 's.png') for f in depthf]
        rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]
        self.stiffness_files = stiffnessf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    # def _get_crop_attrs(self, idx):
    #     gtbbs = grasp.GraspRectangles.load_from_softnet_file(self.grasp_files[idx])
    #     center = gtbbs.center
    #     left = max(0, min(center[1] - self.output_size // 2, 300 - self.output_size))
    #     top = max(0, min(center[0] - self.output_size // 2, 300 - self.output_size))
    #     return center, left, top

    # def get_gtbb(self, idx, rot=0, zoom=1.0):
    #     gtbbs = grasp.GraspRectangles.load_from_softnet_file(self.grasp_files[idx])
    #     center, left, top = self._get_crop_attrs(idx)
    #     gtbbs.rotate(rot, center)
    #     gtbbs.offset((-top, -left))
    #     gtbbs.zoom(zoom, (self.output_size//2, self.output_size//2))
    #     return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.Image.from_file(self.depth_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # depth_img.rotate(rot, center)
        # depth_img.crop((top, left), (min(300, top + self.output_size), min(300, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img
        
    def get_stiffness(self, idx, rot=0, zoom=1.0):
        stiffness_img = image.Image.from_file(self.stiffness_files[idx])
        # print(self.stiffness_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # stiffness_img.rotate(rot, center)
        # stiffness_img.crop((top, left), (min(300, top + self.output_size), min(300, left + self.output_size)))
        stiffness_img.normalise()
        stiffness_img.zoom(zoom)
        stiffness_img.resize((self.output_size, self.output_size))
        return stiffness_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_name(self,idx):
        name = self.stiffness_files[idx].replace('s.png','')
        object_name_stiffness = name[21:] #remove path
        return object_name_stiffness
