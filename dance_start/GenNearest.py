
from turtle import distance
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ generator of image from skeleton """
        closest_dist = float('inf')
        closest_image = None

        # 1. Compute the distance between ske and each skeleton in videoSkeletonTarget
        for i in range(self.videoSkeletonTarget.skeCount()):
            dist = ske.distance(self.videoSkeletonTarget.ske[i])
            if dist < closest_dist:
                closest_dist = dist
                closest_image = self.videoSkeletonTarget.readImage(i)

        if closest_image is not None:
            return closest_image
        else :
            # Return a random image if no valid closest image is found
            return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)



