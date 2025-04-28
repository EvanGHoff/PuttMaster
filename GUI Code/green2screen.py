'''
Author: Manu Ramesh
From ECE661 HW3
'''

import numpy as np, cv2, pdb, copy

def estimateProjRemoverH(locsIn):
    '''
    Author: Manu Ramesh
    gives the H matrix to remove pure projective distortion
    locsIn must be in this order - left top, left bottom, right top, right bottom
    '''
    ltop = locsIn[0].tolist();
    lbot = locsIn[1].tolist();
    rtop = locsIn[2].tolist();
    rbot = locsIn[3].tolist();

    #ltop, lbot, rtop, rbot = locsIn.copy()
    ltop.append(1); lbot.append(1); rtop.append(1); rbot.append(1); #change to HC
    ltop = np.array(ltop); lbot = np.array(lbot); rtop = np.array(rtop); rbot = np.array(rbot)
    #print('ltop = ',ltop, ', lbot = ', lbot, ', rtop = ', rtop, ', rbot = ', rbot)
    
    #to get VP1
    l1 = np.cross(ltop,lbot); l2 = np.cross(rtop,rbot);
    vp1 = np.cross(l1,l2)
    #print('VP1 = ', vp1)
    #to get VP2
    l1 = np.cross(ltop,rtop); l2 = np.cross(lbot,rbot)
    vp2 = np.cross(l1,l2)
    #print("VP2 = ", vp2)
    #to get VL
    vl = np.cross(vp1,vp2)
    vl = vl/vl[2]
    #print("Vanishing line = ", vl)
    #to calculate Hp matrix
    Hp = np.array([[1,0,0], [0,1,0]])
    Hp = np.append(Hp,vl)
    Hp = np.reshape(Hp,(3,3))
    #print('Hp = \n', Hp) 
    #the last index (3,3) will be 1 because we have homogenized vl
   
    return Hp



def calcHomographyAffine(P,Q,R,S):
    '''
    Author: Tejas Pant https://github.com/tejaspant/ECE-661-Computer-Vision/blob/master/HW3/TwoStepMethod_Tejas_Pant.ipynb

    Modified by Manu Ramesh
    Points should be selected in this way
    P-----------Q
    |           |
    |           |
    |           | 
    |           |
    S-----------R
    '''     
    #P = np.array(P); Q = np.array(Q); R = np.array(R); S = np.array(S)

    #pdb.set_trace()


    A = np.zeros((2,2))
    S_mat = np.ones((2,2))
    H = np.identity(3)
    
    #First pair of perpendicular lines
    l1 = np.cross(P,Q) #l'
    m1 = np.cross(Q,R) #m'
    
    #Second pair of perpendicular lines
    l2 = np.cross(R,P)
    m2 = np.cross(S,Q)
    
    A[0,:] = [l1[0]*m1[0], l1[0]*m1[1] + l1[1]*m1[0]]
    A[1,:] = [l2[0]*m2[0], l2[0]*m2[1] + l2[1]*m2[0]]
    
    B = np.array([-l1[1]*m1[1], -l2[1]*m2[1]])
    
    S_elem = np.matmul(np.linalg.pinv(A),B)
    S_mat[0,:] = [S_elem[0], S_elem[1]]
    S_mat[1,0] = S_elem[1]
    
    U, D, Vt = np.linalg.svd(S_mat, full_matrices=True)
    D_sq = np.zeros((2,2))
    D_sq[0,0] = np.sqrt(D[0])
    D_sq[1,1] = np.sqrt(D[1]) 
    
    A_forS = np.matmul(np.matmul(Vt,D_sq),np.transpose(Vt))
    
    H[0:2,0:2] = A_forS

    H = np.linalg.inv(H.astype(np.float32))

    return H

def green2screen(locsIn):
    '''
    Author: Manu Ramesh                
    locsIn must be in this order - left top, left bottom, right top, right bottom
    Example: locsIn = [[478,721], [481,873], [600,739], [605,921]]

    Use the matrix H_g2s to warp the image from green to screen coordinates
    '''
    
    #Remove Projective distortion
    Hp = estimateProjRemoverH(np.array(locsIn))
    
    locsNoProj = np.array(locsIn).T
    locsNoProj = np.append(locsNoProj,np.ones((1,4)),axis=0)
    locsNoProj = np.matmul(Hp,locsNoProj)
    locsNoProj = locsNoProj / locsNoProj[2,:]
    locsNoProj_notHC = locsNoProj[0:2,:].T.tolist()
    locsNoProj = locsNoProj.T.tolist() #Retain HC

    # print(f"\n\nLocs no Proj = {locsNoProj}\n\n")

    #Remove Affine distortion
    #Haf = estimateAffRemoverH(np.array(locsNoProj))

    #pdb.set_trace()
    Haf = calcHomographyAffine(locsNoProj[0], locsNoProj[2], locsNoProj[3], locsNoProj[1])

    # print(f"Haf = {Haf}")

    #pdb.set_trace()
    locsNoProj = [locsNoProj[0], locsNoProj[2], locsNoProj[3], locsNoProj[1]] #change order to Tejas Pant's order
 
    locsNoAf = np.array(locsNoProj).T
    ##locsNoAf = np.append(locsNoAf,np.ones((1,4)),axis=0)
    locsNoAf = np.matmul(Haf,locsNoAf)
    locsNoAf = locsNoAf / locsNoAf[2,:]
    locsNoAf_notHC = locsNoAf[0:2,:].T.tolist()
    locsNoAf = locsNoAf.T.tolist() #preserve HC
    
    # print(f"Locs no AF = {locsNoAf}")
    

    #Order has changed - top left, top right, bottom right, bottom left
    
    #calculate aspect ratio
    P, Q, R, S = locsNoAf
    dist = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    AspectRatio = dist(P, Q) / dist(P, R)
    # print(f"Aspect Ratio = {AspectRatio}")

    
    #find screen coordinates for AR >1, and projector resolution of 1920x1080
    if AspectRatio > 1: #horizontal rectangle
        Ps, Qs, Rs, Ss = [0,0], [1920, 0], [1920, 1080/AspectRatio], [0, 1080/AspectRatio] 
    else: #vertical rectangle, we need to rotate the screen by 90 degrees
        Qs, Rs, Ss, Ps = [0,0], [1920, 0], [1920, 1080*AspectRatio], [0, 1080*AspectRatio] 

    screen_coords = [Ps, Qs, Rs, Ss]
    # print(f"Screen coordinates = {screen_coords}")

    #no Affine to screen
    # pdb.set_trace()
    Hs = cv2.getPerspectiveTransform(np.array(locsNoAf_notHC).astype(np.float32), np.array([Ps, Qs, Rs, Ss]).astype(np.float32))
    # print(f"Hs = {Hs}")


    #the final homography matrix to take points from green to screen
    H_g2s = np.matmul(Haf, Hp)
    H_g2s = np.matmul(Hs, H_g2s)

    # print(f"H_g2s = {H_g2s}")

    #H_g2s = np.linalg.inv(H_g2s) #this is the final homography matrix to take points from green to screen

    # intermediatePTS = np.array(locsNoAf_notHC); H_g2s = np.matmul(Haf, Hp)
    # intermediatePTS = np.array(locsNoProj_notHC); H_g2s = Hp

    

    return H_g2s, screen_coords, AspectRatio

'''
Instructions:
1. Get coordinates of the green.
2. Pass them to the function green2screen(locsIn) in the order of left top, left bottom, right top, right bottom
3. The function will return the homography matrix H_g2s and the screen coordinates in the order of top left, top right, bottom right, bottom left (Order is different!)
4. Use the homography matrix to warp the image of green from camera to a full screen image coordinates using cv2.warpPerspective
6. Crop the image to the screen coordinates.
5. Run the ball and hole detection code on the cropped image. Draw whatever you want on this image (lines, circles, etc.)
6. Stretch the image to fit the screen size of 1920x1080 using cv2.resize.
7. Use the function my warp to warp the full screen image to the projector coordinates and display it on the projector.
'''


    




if __name__ == "__main__":
    
    locs3 = [[2057,700], [2092,1481], [2667,717], [2693,1322]]
    #locs = [[478,721], [481,873], [600,739], [605,921]]
    locs2 = [[478,721], [481,873], [600,739], [605,921]]

    locs = locs3

    #H_g2s, intermediatePTS = green2screen(locs)
    H_g2s, screen_coords = green2screen(locs)
    img = cv2.imread('./Img3.JPG')
    #img = cv2.imread('./Img2.jpeg')

    #print(f"img shape = {img.shape}")

    img_out = cv2.warpPerspective(img, H_g2s, (1920,1080))
    

    # Draw green dots on the image at the positions defined by locs
    for loc in locs:
        cv2.circle(img, (int(loc[0]), int(loc[1])), radius=10, color=(0, 255, 0), thickness=-1)

    # Save the image with the green dots
    cv2.imwrite('input_with_dots.jpg', img)


    #for loc in intermediatePTS:
    #    cv2.circle(img_out, (int(loc[0]), int(loc[1])), radius=10, color=(0, 0, 255), thickness=-1)

    cv2.imwrite('output.jpg', img_out)