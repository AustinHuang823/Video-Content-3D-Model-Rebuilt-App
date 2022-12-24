import cv2 as cv
import os
import numpy as np

def draw_and_save_img(img, p, out_dir,catgories, k):
    for i in range(p.shape[0]):
        pt = p[i].reshape(-1)
        cv.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
    f_name = catgories +'_{}'.format('{0}'.format(str(k).zfill(3))) + '.png'
    cv.imwrite(os.path.join(out_dir, f_name), img)

def find_chain_pass(keypoint, kp_points, z):
    keypoint_list = [keypoint]
    return find_chain(keypoint_list, kp_points, z)

def find_chain(key_point_list, kp_points, z):
    key_point = key_point_list[-1]
    if z == 0:
        return [key_point]
    else:
        temp_list = find_chain([kp_points[z-1][key_point.class_id]], kp_points, z-1)
        for item in temp_list:
            key_point_list.append(item)
        return key_point_list

def g_builder(af, bf):
    return [af[0]*bf[0], 2*af[0]*bf[1], 2*af[0]*bf[2], af[1]*bf[1], 2*af[1]*bf[2], af[2]*bf[2]]

def G_mat_builder(R):
    n = R.shape[0]//2
    G1, G2, G3 = np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 6))
    R1, R2 = R[:n, :], R[n:, :]
    for i in range(n):
        r1 = R1[i, :]
        r2 = R2[i, :]
        gt1, gt2, gt3 = g_builder(r1, r1), g_builder(r2, r2), g_builder(r1, r2)
        G1[i, :], G2[i, :], G3[i, :] = gt1, gt2, gt3

    G = np.vstack((G1,G2,G3))    

    return G
