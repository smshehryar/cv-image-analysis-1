from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import os 


# Function to Convert Images to grayscale
def ICV_grayscale(img):

    height, width = img.shape[:2]     
    
    imgOutput= np.zeros(shape=(height,width))
    for i in range(height):
        for j in range(width):
           imgOutput[i, j] = (int(img[i, j, 0]*0.299)+int(img[i, j, 1]*0.587)+int(img[i, j, 2]*0.114))
           
    #plt.imshow(imgOutput)
    #plt.show()
    return imgOutput
#ICV_grayscale("Dataset/DatasetA/car-1.JPG")


############################################################
# START Question - 1 - Transformations
############################################################

def ICV_transform_Image(fileName, conv, rotAngle, skewAngle):
    
    thetaShear = np.radians(skewAngle) # angle to shear 
    thetaRot = np.radians(rotAngle)# angle to rotate

    cos1 = (np.cos(thetaRot))# calculating cosine for rotation
    sin1 = (np.sin(thetaRot))# calculating sine for rotation
    shearX = ((np.tan(thetaShear)))# calculating shear angle
    #shearY = (1/(np.tan(thetaShear)))
    
    # Rotation Matrix
    Tr = np.array([
        [cos1,sin1],
        [-sin1,cos1]])

    # Skew Matrix    
    Tsk = np.array([
        [1,-shearX],
        [0,1]]).T
    Ts = np.array([
        [2,0],
        [0,2]])

    # Combined Matrix i
    Tcomb1 = Tsk@Tr

    # Combined Matrix ii
    Tcomb2 = Tr@Tsk
    
    

    if(conv == 0):
        Tf = Tsk
    if(conv == 1):
        Tf = Tr
    if(conv == 2):
        Tf = Tcomb1
    if(conv == 3):
        Tf = Tcomb2



    img = plt.imread(fileName)

    height, width = img.shape[:2]

    main_dimension = height
    if(height>=width):
        main_dimension = height
    else:
        main_dimension = width
    
    center_x = main_dimension
    center_y = main_dimension
    imgOutput = np.ndarray((main_dimension*2,main_dimension*2,3))
    
    #loop over the entire image
    for i in range(height):
        for j in range(width):
            pixelValue = img[i, j, :]
            
            # positioning relative to the center of the input image
            i_transfO = i-(height/2)
            j_transfO = j-(width/2)


            #transforming image coordinates
            i_transf = i_transfO*Tf[0,0] + Tf[0,1]*j_transfO
            j_transf = i_transfO*Tf[1,0] + Tf[1,1]*j_transfO

            #centering image coordinates back relative to the output image
            i_transf+= center_x
            j_transf+=center_y
            # the transformed coordinates are used for the new image
            if(0<i_transf < main_dimension*2 and 0<j_transf < main_dimension*2):
                imgOutput[int(i_transf), int(j_transf), :] = pixelValue

                
    plt.imshow(imgOutput.astype('uint8'))
    plt.show()

    if(conv>0):
        imgOutputInv = np.ndarray((main_dimension*2,main_dimension*2,3))
        center_x = height/2
        center_y = width/2
        TInv = np.linalg.inv(Tf)#computing the inverse matrix

        for i in range(height):
            for j in range(width):
                
                i_transfI = i-center_x
                j_transfI = j-center_y
                
                inputCoords= np.array([i_transfI, j_transfI])
                i_transfI, j_transfI = np.dot(TInv,inputCoords)
                
                i_transfI+= center_x
                j_transfI+=center_y
                # the inverse transformed coordinates are used for the original image this time
                if(0<i_transfI < height and 0<j_transfI < width):
                    imgOutputInv[int(i+main_dimension/2), int(j+main_dimension/2), :] = img[int(i_transfI), int(j_transfI), :]


        plt.imshow(imgOutputInv.astype('uint8'))
        plt.show()
    

#ICV_transform_Image("name_image.JPG", 1, -50, 60)
ICV_transform_Image("Dataset/DatasetA/car-1.JPG", 1, 20, 50)


def ICV_createTextImage():

    imgOutput = Image.new('RGB', (400, 100), color = 'blue')
    #rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
    fnt = ImageFont.truetype('arial.ttf', 72)
    draw = ImageDraw.Draw(imgOutput)
    draw.text((0,0), "SHEHRYAR", font=fnt,fill=(255,255,255))
    
    imgOutput.save('name_image.jpg')
    plt.imshow(imgOutput)
    plt.show()
#ICV_createTextImage()



############################################################
# END Question - 1 - Transformations
############################################################




############################################################
# START Question - 2 - Convolutions
############################################################



# gray = 1 or 0, filter = 0,1,2 for average, A and B respectively
def ICV_filter_Image(fileName, gray, filter):
    
    #Mean filter
    filterAvg = np.array([
        [1/9,1/9,1/9],
        [1/9,1/9,1/9],
        [1/9,1/9,1/9]])
    
    #Laplacian filter
    filterB = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]])
    
    # Gaussian blur
    filterA = np.array([
        [1/16,2/16,1/16],
        [2/16,4/16,2/16],
        [1/16,2/16,1/16]])

    if(filter == 0):
       Tfilter = filterAvg
    elif(filter == 1):
       Tfilter = filterA
    elif(filter == 2):
       Tfilter = filterB
    
    img = plt.imread(fileName)
    if(gray == 1):
        img = ICV_grayscale(img)

    height, width = img.shape[:2]

    imgOutput = np.ndarray((height,width,3))
    
    upper = 0
    lower = 0
    right = 0
    left = 0

    # if image is grayscale
    if(gray == 1):
        # iterating over the whole image
        for i in range(height):
            for j in range(width):
                
                # conditional statements to ignore the borders
                if(i-1>height):
                    upper = img[i-1, j]*(Tfilter[0,1])

                if(i+1<height):
                    lower = img[i+1, j]*(Tfilter[2,1])
                    lower_left = img[i+1, j-1]*(Tfilter[2,0])

                if(j+1<width):
                    right = img[i, j+1]*(Tfilter[1,2])
                    upper_right= img[i-1, j+1]*(Tfilter[0,2])
                
                if(j+1<width) and (i+1<height):
                    lower_right = img[i+1, j+1]*(Tfilter[2,2])

                left = img[i, j-1]*(Tfilter[1,0])
                upper_left = img[i-1, j-1]*(Tfilter[0,0])
                
                # the center value is computed by also adding the filtered values of the neighbouring pixels
                imgOutput[i, j] = (img[i, j]*(Tfilter[1,1]) + upper+lower+left+right+lower_left+lower_right+upper_left+upper_right)
                
                # each newly assigned pixel value is checked for an overflow of value
                for channel in range(3):
                    if(imgOutput[i, j, channel] > 255):
                        imgOutput[i, j, channel] = 255
                    if(imgOutput[i, j, channel] < 0):
                        imgOutput[i, j, channel] = 0

    # if image is not grayscale
    else:

        for i in range(height):
            for j in range(width):
                
                
                if(i-1>height):
                    upper = img[i-1, j, :]*(Tfilter[0,1])

                if(i+1<height): 
                    lower = img[i+1, j, :]*(Tfilter[2,1])
                    lower_left = img[i+1, j-1, :]*(Tfilter[2,0])

                if(j+1<width):
                    right = img[i, j+1, :]*(Tfilter[1,2])
                    upper_right= img[i-1, j+1, :]*(Tfilter[0,2])
                
                if(j+1<width) and (i+1<height):
                    lower_right = img[i+1, j+1, :]*(Tfilter[2,2])

                left = img[i, j-1, :]*(Tfilter[1,0])
                upper_left = img[i-1, j-1, :]*(Tfilter[0,0])
                

                imgOutput[i, j, :] = (img[i, j, :]*(Tfilter[1,1]) + upper+lower+left+right+lower_left+lower_right+upper_left+upper_right)
                
                
                for channel in range(3):
                    if(imgOutput[i, j, channel] > 255):
                        imgOutput[i, j, channel] = 255
                    if(imgOutput[i, j, channel] < 0):
                        imgOutput[i, j, channel] = 0

            
    plt.axis('off')
            
    plt.imshow(imgOutput.astype('uint8'))
    plt.savefig("test.jpg", bbox_inches='tight') 
    plt.show()

#ICV_filter_Image("test.JPG",1,2)
#ICV_filter_Image("Dataset/DatasetA/car-1.JPG",1,2)

############################################################
# END Question - 2 - Convolutions
############################################################



############################################################
# START Question - 3 - Histograms and Video Segmentation
############################################################

# intersection = 1 or 0
def ICV_histogram_Image(imagehist, intersection):

    img = plt.imread(imagehist)
    imgGray = ICV_grayscale(img)
    

    height, width = img.shape[:2]
    
    #no_pixels_channel = width*height

    
    #create integer arrays for 256 values
    red_hist = [0]*256
    green_hist = [0]*256
    blue_hist = [0]*256
    final_hist = [0]*256

    
    #iterate over the whole image for image values
    for i in range(height):
        for j in range(width):
            # use the image pixel value as the index value of each channel's histogram and increment the indexed value of that channel histogram 
            final_hist[int(imgGray[i, j])] += 1

    
    # if intersection is not required also compute the histograms of the channels separately
    if(intersection == 0):
        #iterate over the whole image for image values
        for i in range(height):
            for j in range(width):
                # use the image pixel value as the index value of each channel's histogram and increment the indexed value of that channel histogram 
                red_hist[int(img[i, j,0])] += 1
                green_hist[int(img[i, j, 1])] += 1
                blue_hist[int(img[i, j, 2])] += 1

        #graph for color channels histogram
        for i in range(0, 256):
            
            #normalize the histogram values
            red_hist[i]/=1
            green_hist[i]/=1
            blue_hist[i]/=1
            
            plt.bar(i, red_hist[i], color='red', alpha=0.6)
            plt.bar(i, green_hist[i], color='green', alpha=0.6)
            plt.bar(i, blue_hist[i], color='blue', alpha=0.8)

        plt.xlabel('Pixel Range (0-255)')
        plt.ylabel('Pixel Frequencies')
        plt.title('RGB Histogram')
        plt.show()

        # graph the grayscale histogram
        for i in range(0, 256):
            plt.bar(i, final_hist[i], color='black', alpha=0.8)

    
        plt.xlabel('Pixel Range (0-255)')
        plt.ylabel('Pixel Frequencies')
        plt.title('Grayscale Histogram')
        plt.show()
    
   
    return final_hist, np.sum(final_hist)

#ICV_histogram_Image("cap1.JPG",0)


# seq_intersection = 0 or 1, normalize = 0 or 1
# function takes the intersection between the histograms of two images, and returns the overlap value
def ICV_histogram_intersection(img_1, img_2, seq_intersection, normalize):
    
    sum_hist = 0.0
   
    # getting the histograms of the two frames
    hist_1, no = ICV_histogram_Image(img_1,1)
    hist_2, no_pixels = ICV_histogram_Image(img_2,1)
    inter_hist = [0]*256

    # to calculate the intersection value, take the minimum of value amongst the two histograms.
    # Also calculate the sum of the intersection values
    # If required to normalize divide by the number of pixels
    for i in range(0,256):
        inter_hist[i] = min(hist_1[i], hist_2[i])
        if(normalize == 1):
            inter_hist[i] /= no_pixels
        sum_hist += inter_hist[i]

    # check whether if it is a video sequence, plotting individual intersections would not be efficient
    if(seq_intersection == 0):
        #plotting the values of the two histograms together
        for i in range(0,256):
            
            plt.bar(i, hist_1[i], color='gray', alpha=0.8)
            plt.bar(i, hist_2[i], color='yellow', alpha=0.5)

        plt.xlabel('Pixel Range (0-255)')
        plt.ylabel('Pixel Frequencies')
        plt.title('Overlap of the two histograms')
        plt.show()

        #plotting the intersection values of the histograms
        for i in range(0,256):
            plt.bar(i, inter_hist[i], color='black', alpha=0.8)

        plt.xlabel('Pixel Range (0-255)')
        plt.ylabel('Pixel Frequencies')
        plt.title('Plot for intersection values')
        plt.show()

    
    
    print("Intersection Value: ", (sum_hist))
    if(normalize == 1):
        print("Normalized Intersection Value: ", (sum_hist)*100)

    return (sum_hist)
#ICV_histogram_intersection("cap0.JPG","cap1.JPG",0, 0)    

# normalize = 1 and 0
def ICV_capVid(filename, normalize):
    cam = cv2.VideoCapture(filename) 
    inter_vals = [0]*int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    i=0

    if(normalize ==1):
        max_val = 1.0 # for normailzed value between 0 and 1
    else:
        max_val = 80000

    while(cam.isOpened()):
        ret, frame = cam.read()
        if ret == False:
            break
        cv2.imwrite('cap'+str(i)+'.jpg',frame)
        
        # only if one frame has been captured, then start with taking intersections
        if(i>0):
            img_1 = 'cap'+str(i-1)+'.jpg'
            img_2 = 'cap'+str(i)+'.jpg'
            inter_vals[i] = ICV_histogram_intersection(img_1, img_2,1,normalize)
            sec = ((int(cam.get(cv2.CAP_PROP_POS_MSEC)))/1000)# convert milliseconds to seconds
            print(i, sec, img_1, img_2, inter_vals[i]) 
            plt.bar(sec, (max_val-inter_vals[i]), color='black', alpha=0.8)
        i+=1

    plt.xlabel('Time (seconds)')
    plt.ylabel('Intersection Values (max intersection value-intersection value)')    
    plt.show()
    cam.release()
    cv2.destroyAllWindows()

#ICV_capVid("Dataset/DatasetB.AVI",1)

############################################################
# END Question - 3 - Histograms and Video Segmentation
############################################################

############### References
#https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/

############### README

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