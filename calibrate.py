import pickle


# src
display_markers()
src_points = detect_aruco_markers(1)
pickle.dump(src_points, open('Raspberry PI Code/matrixes/srcPts.p','wb'))

# dst
rectangle()
cv2.waitKey(1)
dst_points = detect_green_rectangle()
pickle.dump(dst_points, open('Raspberry PI Code/matrixes/dstPts.p','wb'))
# input() #pause to find points

matrix = cv2.getPerspectiveTransform(src_points, dst_points)
pickle.dump(matrix, open('Raspberry PI Code/matrixes/matrix.p','wb'))