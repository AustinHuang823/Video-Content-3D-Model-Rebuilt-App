import cv2 as cv
import os
import Functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

"""
using SIFT from cv2 to retain features
nfeatures: The number of best features to retain
The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
"""
nBF = 2000
sift = cv.SIFT_create(nfeatures=nBF)

# FlannBasedMatcher from cv2 trains cv::flann::Index on a train descriptor collection 
# and calls its nearest search methods to find the best matches.
flann = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

dataset_name = ['Medusa Head', 'Castle Sequence']
dir_name = ['medusajpg', 'castlejpg']
first_img = ['medusa_out0001.jpg','castle.000.jpg']
in_f_name = ['medusa_out0{}.jpg','castle.{}.jpg']
out_dir = ['medusaout', 'castleout']
match_out_dir = ['medusamatch','castlematch']
catgories = ['medusa_out', 'castle_out']
img_num = [50, 27]


"""
Perform the followings for each dataset
"""

for i in range(0,len(dir_name)):
    # Use the first image in each dataset to set up first data
    original_RGBimg = cv.imread(os.path.join(dir_name[i],first_img[i]))
    # roi = cv.selectROI(original_RGBimg)
    # original_RGBimg = original_RGBimg[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    original_GRAYimg = cv.cvtColor(original_RGBimg, cv.COLOR_BGR2GRAY)

    kp_points, mp_points, good_kp_points = [], [], []
    # Detect keypoints and computes their descriptors
    kp0, des0 = sift.detectAndCompute(original_GRAYimg, None)
    kp_points.append(kp0)

    # Convert vector of keypoints to vector of points or vise versa
    p0 = cv.KeyPoint_convert(kp0)
    Functions.draw_and_save_img(original_RGBimg, p0, out_dir[i], catgories[i], 0)
    
    last_img = original_GRAYimg
    last_kp = kp0
    for j in range(1, img_num[i]+1):
        f_name = os.path.join(dir_name[i], in_f_name[i].format('{0}'.format(str(j).zfill(3))))
        
        RGB_img = cv.imread(f_name)
        # RGB_img = RGB_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        GRAY_img = cv.cvtColor(RGB_img, cv.COLOR_BGR2GRAY)

        kpj, desj = sift.detectAndCompute(GRAY_img, None)
        kp_points.append(kpj)

        # Find the k best matches for each descriptor from a query set
        des0 = np.float32(des0)
        desj = np.float32(desj)
        knn_matches = flann.knnMatch(des0, desj, k = 2)

        # Filter matches using the Lowe's distance ratio test
        ratio_thresh = 0.7
        good_matches = []
        match_kp_points = []
        for idz in range(len(knn_matches)):
            m,n = knn_matches[idz]
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                match_kp_points.append(kp_points[j][idz])
        good_kp_points.append(match_kp_points)

        # Draw matches
        img_matches = np.empty((max(last_img.shape[0], GRAY_img.shape[0]), last_img.shape[1]+GRAY_img.shape[1], 3), dtype=np.uint8)
        cv.drawMatches(last_img, last_kp, GRAY_img, kpj, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # Show detected matches
        cv.imshow('Good Matches', img_matches)
        match_f_name = catgories[i][:-4] + '_match' + '_{}'.format('{0}'.format(str(j).zfill(3))) + '.png'
        cv.imwrite(os.path.join(match_out_dir[i], match_f_name), img_matches)
        cv.waitKey(10)

        mp_points.append(good_matches)

        pj = cv.KeyPoint_convert(kpj)
        Functions.draw_and_save_img(RGB_img, pj, out_dir[i], catgories[i], j)

        des0, last_img, last_kp = desj, GRAY_img, kpj
    
    MinNumOfGdKP = float('inf')
    for n in range(len(good_kp_points)):
        MinNumOfGdKP = min(len(good_kp_points[n]), MinNumOfGdKP)
    
    SameLengthKP_points = []
    for nFrame in range(len(good_kp_points)):
        temp = []
        for idx2MinNum in range(MinNumOfGdKP):
            temp.append(good_kp_points[nFrame][idx2MinNum])
        SameLengthKP_points.append(temp)

    chain_list = []
    for n in range(len(SameLengthKP_points[0])):
        temp = []
        for m in range(len(SameLengthKP_points)):
            temp.append(SameLengthKP_points[m][n])
        chain_list.append(temp)

    img_points = []
    for c in chain_list:
        c_points = cv.KeyPoint_convert(c)
        img_points.append(c_points)

    img_points = np.array(img_points)

    """
    The following process follows the Tomasi-Kanade Factorization algorithm in 
    "Shape and Motion from Image Streams under Orthography: a Factorization Method"
    """

    # Number of feature points: P
    feature_points = img_points.shape[0]
    # Number of frames in data: F (+1 original frame)
    n_img = img_points.shape[1]

    points = np.swapaxes(img_points, 2, 0)

    # Matrices U,V (size FxP)
    U, V = points[0],points[1]
    # Measurement matrix W: stacking U and V (size 2FxP)
    W = np.vstack((U,V))
    # Registered measurement matrix ~W: (size 2FxP)
    W = W - np.mean(W, axis = 1)[:,None]
    
    # Approximate Rank
    # Determine if 2F>=P or 2F<P
    Transpose = False
    if W.shape[0] < W.shape[1]:
        W = W.T
        Transpose = True

    # Decompose ~W using SVD
    # O1 (size 2FxP), \Sigma (size PxP), O2 (size PxP)
    O1, S, O2 = np.linalg.svd(W, full_matrices=False)
    
    # Rank Theorem for Noisy Measurements
    # consider only the three greatest singular values of ~W
    # with the corresponding left and right eigenvectors
    S_psqrt = np.diag(np.sqrt(S[:3]))
    if Transpose:
        O1, O2 = O2.T, O1.T   
    O1_p, O2_p = O1[:,:3], O2[:3,:]
    # W^ = R^S^
    R_hat, S_hat = np.dot(O1_p, S_psqrt), np.dot(S_psqrt, O2_p)
    
    """
    The following process follows the steps in
    "A Sequential Factorization Method for Recovering Shape and Motion from Image Streams"
    to compute R and S
    """

    # The Metric Constraints
    # Gl = c
    # G (size 3Fx6), l (size 6x1), c (size 3Fx1)
    G = Functions.G_mat_builder(R_hat)
    c1 = np.ones((2*n_img, 1))
    c2 = np.zeros((n_img, 1))
    c = np.vstack((c1, c2)).squeeze()
    GTG_inv = np.linalg.pinv(np.dot(G.T, G))
    l = np.dot(np.dot(GTG_inv, G.T), c)
    
    # Construct the symmetric matrix L from the vector l
    # L's eigendecomposition gives an affine transformation matrix A
    L = np.zeros((3,3))
    L[0, 0], L[1, 1], L[2, 2] = l[0], l[3], l[5]
    L[0, 1], L[0, 2], L[1, 2] = l[1], l[2], l[4]
    L[1, 0], L[2, 0], L[2, 1] = l[1], l[2], l[4]

    # Enforce Positive Semi-Definite
    e_vl, e_vt = np.linalg.eig(L)
    D = np.diag(e_vl)
    D[D < 0] = 0.000001
    L = np.dot(e_vt, np.dot(D, e_vt.T))

    # Obtain the invertible matrix Q by Cholesky Decomposition
    Q = np.linalg.cholesky(L)

    # Compute True rotation matrix R and True shape matrix S
    R_true = np.dot(R_hat, Q)
    S_true = np.dot(np.linalg.inv(Q), S_hat)
    x,y,z = S_true[0,:], S_true[1,:], S_true[2,:]
    
    # Datatype setup for plotting
    factor = 1
    S_pt = np.zeros((x.shape[0], 3))
    S_pt[:, 0] = x
    S_pt[:, 1] = y
    S_pt[:, 2] = z * factor
    xs = S_pt[:, 0]
    ys = S_pt[:, 1]
    zs = S_pt[:, 2]

    """
    Plot Results
    """

    fig = plt.figure()
    fig.set_size_inches(15, 7.5, forward=True)
    fig.suptitle(f'Reconstructed {dataset_name[i]} 3D Object using Tomasi-Kanade Factorization Method', 
                c='midnightblue', size=16 ,fontweight='bold')
    ax1 = fig.add_subplot(121, projection='3d', 
                            xlabel=r'$x$', ylabel=r'$y$', zlabel=r'$z$')
    ax2 = fig.add_subplot(122, projection='3d', 
                            xlabel=r'$x$', ylabel=r'$y$', zlabel=r'$z$')
    ax1.scatter(x, y, z, c = z, marker = "o", s = 2)
    ax2.scatter(x, y, z, c = z, marker = "o", s = 2)
    # ax1.view_init(azim=-135, elev=60)
    ax2.view_init(azim=-90, elev=90)
    plt.savefig(dir_name[i][:-3]+"plot.png")

    # ax3 = fig.add_subplot(121, projection='3d')
    # pnt3d = ax3.scatter(xs, ys, zs, c=zs, marker="x")
    # X = np.arange(min(x), max(x), 0.2)
    # Y = np.arange(min(y), max(y), 0.2)
    # X, Y = np.meshgrid(X, Y)
    # Z = griddata((x,y),z,(X,Y), method='cubic')
    # ax3.plot_surface(X,Y,Z)
    # ax3.view_init(azim=-135, elev=60)

    plt.show()
