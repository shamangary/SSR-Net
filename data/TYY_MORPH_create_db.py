import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import sys
import dlib
from moviepy.editor import *

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def get_landmarks(im,detector,predictor):
    rects = detector(im, 1)

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str,
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    img_size = args.img_size

    mypath = './morph2'
    isPlot = False
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("landmarks/shape_predictor_68_face_landmarks.dat")
    
    ref_img = cv2.imread(mypath+'/009055_1M54.JPG')
    landmark_ref = get_landmarks(ref_img,detector,predictor)
    
    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))

    # Points used to line up the images.
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                                   RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


    out_genders = []
    out_ages = []
    out_imgs = []

    for i in tqdm(range(len(onlyfiles))):

        img_name = onlyfiles[i]
        temp_name = img_name.split('_')
        temp_name = temp_name[1].split('.')
        isMale = temp_name[0].find('M')
        isFemale = temp_name[0].find('F')
        
        if isMale > -1:
            gender = 0
            age = temp_name[0].split('M')
            age = age[1]
        elif isFemale > -1:
            gender = 1
            age = temp_name[0].split('F')
            age = age[1]

        age = int(float(age))

  
        
        input_img = cv2.imread(mypath+'/'+img_name)
        img_h, img_w, _ = np.shape(input_img)

        
        detected = detector(input_img,1)
        if len(detected) == 1:

            #---------------------------------------------------------------------------------------------
            # Face align

            landmark = get_landmarks(input_img,detector,predictor)
            M = transformation_from_points(landmark_ref[ALIGN_POINTS], landmark[ALIGN_POINTS])
            input_img = warp_im(input_img, M, ref_img.shape)

            #---------------------------------------------------------------------------------------------

            detected = detector(input_img, 1)
            if len(detected) == 1:
                faces = np.empty((len(detected), img_size, img_size, 3))
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - 0.4 * w), 0)
                    yw1 = max(int(y1 - 0.4 * h), 0)
                    xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                    yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                    faces[i,:,:,:] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                
                    if isPlot:
                        cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 255, 0), 2)
                        img_clip = ImageClip(input_img)
                        img_clip.show()
                        key = cv2.waitKey(1000)
                    
                #only add to the list when faces is detected
                out_imgs.append(faces[0,:,:,:])
                out_genders.append(int(gender))
                out_ages.append(int(age))

    np.savez(output_path,image=np.array(out_imgs), gender=np.array(out_genders), age=np.array(out_ages), img_size=img_size)

if __name__ == '__main__':
    main()
