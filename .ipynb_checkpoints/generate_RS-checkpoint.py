import numpy as np
from glob import glob
from scipy import ndimage, misc
import cv2
from PIL import Image, ImageDraw
import re
import pydicom as dicom
from pydicom.sequence import Sequence
import copy

def binary(mask, th):
    mask[mask >= th] = 1
    mask[mask < th]  = 0
    return mask

def check_arrays(array1, array2):
    exists = False
    for i in array1:
        if np.sum(i == array2) > 0:
            exists = True
    return exists

def append_arrays(array1, array2):
    array = np.unique(np.append(array1, array2))
    return array

def distance(point, arrays):
    return np.sqrt((point[0] - arrays[:, 0]) ** 2 + (point[1] - arrays[:, 1]) ** 2)

def search_nearby(current, points, radius):
    cp_distance = distance(current, points)
    tmp_idx = (np.where(cp_distance <= radius))
    return points[tmp_idx]

def find_path(points):
    point_array = np.array([])
    rm_idx = np.random.randint(len(points))
    current = points[rm_idx]
    points = np.delete(points, rm_idx, 0)
    empty = (len(points) == 0)
    point_array = np.append(point_array, current)
    radius_2 = 1 # sqaured radius
    while not empty:
        nearby_points = search_nearby(current, points, 1)
        while len(nearby_points) == 0: # no points nearby in that radius
            radius_2 += 1
            nearby_points = search_nearby(current, points, np.sqrt(radius_2))
        radius_2 = 1

        sampled_idx = np.random.randint(len(nearby_points))
        sampled_point = nearby_points[sampled_idx]
        rm_idx = np.where(np.all(points == sampled_point, axis = 1))
        points = np.delete(points, rm_idx, 0)
        current = sampled_point
        point_array = np.append(point_array, current)
        empty = (len(points) == 0)
    return point_array

def determine_direction(sort_points, is_positive):
    sum_line_integral = 0
    for i in range(len(sort_points) - 1):
        x_1 = sort_points[i, 0]
        x_2 = sort_points[i+1, 0]
        y_1 = sort_points[i, 1]
        y_2 = sort_points[i+1, 1]
        line_integral = (y_2 - y_1) * (x_2 + x_1)
        sum_line_integral += line_integral
    if sum_line_integral < 0:
        clock_wise = 0
    else:
        clock_wise = 1
    if not is_positive ^ clock_wise:
        sort_points = np.flip(sort_points, 0)
    return sort_points

def generate_points(npy):
    mask = binary(npy[:, 1024:], 0.5)
    mask = binary(ndimage.median_filter(mask, size = 2), 0.5)
    edges = np.zeros((512, 512))
    sparse_edges = np.zeros((512, 512))
    for i in range(1, 511):
        for j in range(1, 511):
            if (np.sum(mask[i-1:i+2, j-1:j+2]) > 2) & (np.sum(mask[i-1:i+2, j-1:j+2]) < 6):
                edges[i, j] = 1
            if (np.sum(mask[i-1:i+2, j-1:j+2]) == 5):
                sparse_edges[i, j] = 1

    idx_i, idx_j = (np.where(edges == 1))
    bm_coor = np.transpose(np.array([idx_j, idx_i]))
    sp_idx_i, sp_idx_j = (np.where(sparse_edges == 1))
    sparse_coor = np.transpose(np.array([sp_idx_j, sp_idx_i]))
    tmp_idx_array = []
    curv_points = []

    for i in range(0, len(bm_coor)):
        coor_dist = np.sqrt((bm_coor[:, 0] - bm_coor[i, 0])**2 + (bm_coor[:, 1] - bm_coor[i, 1])**2)
        tmp_idx = np.where(coor_dist <= 2)
        tmp_idx = tmp_idx[0]
        tmp_idx_array.append(tmp_idx)
    coor_empty = False
    del_idx = np.array([])
    curv_points.append(tmp_idx_array[0])
    tmp_idx_array = tmp_idx_array[1:]
    prev_len = 0

    while not coor_empty:
        for it in range(len(tmp_idx_array)):
            for jt in range(len(curv_points)):
                if check_arrays(curv_points[jt], tmp_idx_array[it]):
                    curv_points[jt] = append_arrays(curv_points[jt], tmp_idx_array[it])
                    del_idx = np.append(del_idx, it)
        tmp_idx_array = np.delete(tmp_idx_array, del_idx, 0).tolist()
        del_idx = np.array([])
        if len(tmp_idx_array) == 0:
           coor_empty = True
        elif prev_len == len(tmp_idx_array):
            curv_points.append(tmp_idx_array[0])
            tmp_idx_array = np.delete(tmp_idx_array, 0, 0).tolist()
        prev_len = len(tmp_idx_array)
    new_mask = np.zeros((512, 512))
    refined_Contour_data = []
    for num_p in range(len(curv_points)):
        raw_points = np.transpose(np.array([idx_j[curv_points[num_p]], idx_i[curv_points[num_p]]]))
        sort_points = np.reshape(find_path(raw_points), (-1, 2))
        img = Image.new('L', (512, 512))
        draw = ImageDraw.Draw(img)
        points = []
        coor = sort_points
        for i in range(0, len(coor)):
            points.append(tuple(coor[i]))
        points = tuple(points)
        draw.polygon((points), fill = 1)
        img = np.array(img)
        if np.max(new_mask + img) == 1:
            new_mask += img
            is_positive = True
        else:
            new_mask -= img
            is_positive = False
        direction_coor = determine_direction(sort_points, is_positive)
        refined_coor = direction_coor[np.sum(np.isin(direction_coor, sparse_coor), axis = 1) == 2]
        if len(refined_coor) == 0:
            refined_Contour_data.append(direction_coor)
        else:
            refined_Contour_data.append(refined_coor)
    return refined_Contour_data

if __name__ == "__main__":
    annotation = open("/data/YH/2d_data/patient/annotation.txt", "r")
    info = np.array(annotation.readlines()[1:])
    PID_info = {}
    for i in range(len(info)):
        PID = info[i].split(",")[0]
        bladder = int(info[i].split(",")[1])
        rectum = int(info[i].split(",")[2])
        prostate = int(info[i].split(",")[3])
        RS_info = {}
        RS_info['bladder'] = bladder
        RS_info['rectum'] = rectum
        RS_info['prostate'] = prostate
        PID_info[PID] = RS_info

    path = "/data/YH/2d_data/patient/"
    test = np.array(['44499167', '44704357'])
#    for PID in PID_info.keys():
    for PID in test:
        prostate = PID_info[PID]['prostate'] - 1
        rectum = PID_info[PID]['rectum'] - 1
        bladder = PID_info[PID]['bladder'] - 1
        organ_list = np.array([bladder, prostate, rectum])

        RS_file_name = PID + "/C1/RS*"
        RS = dicom.read_file(glob(path + RS_file_name)[0])
        CP_S = RS.ROIContourSequence
        ref = dicom.read_file(glob(path + PID + "/C1/CT*.dcm")[0])
        ref_coor = np.array(ref.ImagePositionPatient)
        pixel_resol = ref.PixelSpacing[0]
        for organ in organ_list:
            new_Sequence = []
            if organ == prostate:
                npys = sorted(glob('outs/fold_BrewNet_v2_prostate/PID_'+ PID +'*.npy'))
                print("prostate: ", organ)
            elif organ == rectum:
                npys = sorted(glob('outs/fold_BrewNet_v2_rectum/PID_' + PID + '*.npy'))
                print("rectum: ", organ)
            elif organ == bladder:
                npys = sorted(glob('outs/fold_BrewNet_v2_bladder/PID_' + PID + '*.npy'))
                print("bladder: ", organ)
            reference_list= np.array([])
            for i in range(len(CP_S[organ].ContourSequence)):
                UID = CP_S[organ].ContourSequence[i].ContourImageSequence[0].ReferencedSOPInstanceUID
                reference_list = np.append(reference_list, UID)

            for npy_file in npys:
                npy = np.load(npy_file)
                print(npy_file)
                refined_Contour_data = generate_points(npy)
                CT_code = re.split("_|.npy", npy_file)[5]
                is_refslide = np.array([])
                for i in range(len(reference_list)):
                    UID = reference_list[i]
                    is_refslide = np.append(is_refslide, UID.find(CT_code) != -1 )
                ref_slide_idx = np.where(is_refslide == True)[0]
                for i in range(len(refined_Contour_data)):
                    new_Data = copy.deepcopy(CP_S[organ].ContourSequence[ref_slide_idx[0]])
                    xy_ = refined_Contour_data[i]
                    xy_ = np.around(xy_ * pixel_resol + ref_coor[0:2], decimals = 2)
                    #print(xy_)
                    num_points = len(xy_)
                    z_ = copy.deepcopy(CP_S[organ].ContourSequence[ref_slide_idx[0]].ContourData[2])
                    z_ = np.repeat(z_, num_points)
                    z_ = np.reshape(z_, (-1, 1))
                    CT_ID = copy.deepcopy(CP_S[organ].ContourSequence[ref_slide_idx[0]].ContourImageSequence[0].ReferencedSOPInstanceUID)
                    contour_data_ = np.concatenate((xy_, z_), axis = 1)
                    contour_data_ = np.reshape(contour_data_, 3 * num_points).astype(str).tolist()

                    new_Data.ContourData = contour_data_
                    new_Data.NumberOfContourPoints = str(num_points)
                    new_Data.ContourImageSequence[0].ReferencedSOPInstanceUID = CT_ID
                    new_Sequence.append(new_Data)
            CP_S[organ].ContourSequence = new_Sequence
        RS.ROIContourSequence = CP_S
        new_file_name = glob(path + RS_file_name)[0].split(".dcm")[0] + "_modified.dcm"
        dicom.filewriter.write_file(new_file_name, RS)
