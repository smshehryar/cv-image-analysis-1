############### README
To run a function please uncomment it in the Python file. All functions with the readme are also given at the end 
of the Python file. 
############### TRANSFORMATIONS
############################################################
#ICV_transform_Image(filename, transformation, rotation angle, skew angle)
#transformations: 0=skew matrix, 1=rotation matrix, 2=rotation+skew, 3=skew+rotation

#ICV_transform_Image("Dataset/DatasetA/car-1.JPG", 3, 20, 50)

############### CONVOLUTIONS
############################################################
#ICV_filter_Image(filname,convert it to grayscale?,filter)
#filter: 0=average filter, 1=filter A, 2=filter=B
#convert it to grayscale?: 0=No, 1=Yes

#ICV_filter_Image("Dataset/DatasetA/car-1.JPG",1,2)

############### HISTORGRAMS AND VIDEO SEGMENTATION
############################################################
#ICV_histogram_Image(filename,is for intersection?)
#is for intersection?: 0=No, 1=Yes 

#ICV_histogram_Image("cap1.JPG",0)


#ICV_histogram_intersection(filename1,filename2, is video sequence?, to normalize?)
#is video sequence?: 0=No, 1=Yes
#to normalize?: 0=No, 1=Yes

#ICV_histogram_intersection("cap0.JPG","cap1.JPG",0, 0)


#ICV_capVid(filename, to normalize?)
#to normalize?: 0=No, 1=Yes

#ICV_capVid(filename,1) 