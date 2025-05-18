import os
import matplotlib.pyplot as plt
import meerkat
import cv2
import numpy as np
import pickle

def skin_pixel_identification(image):
        img_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # skin color range for hsv color space
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # converting from gbr to YCbCr color space
        img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # skin color range for hsv color space
        YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
        YCrCb_mask = cv2.morphologyEx(
            YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )

        # merge skin detection (YCbCr and hsv)
        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(
            global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8)
        )

        masked_img = cv2.bitwise_and(image, image, mask=global_mask)
        return masked_img


def generate_plot_images(mktfile, posefile):
    reader = meerkat.MktReader(
        mktfile,
        auto_fix_timestamp=True,
    )
    colour, depth, ir, params = reader.read_frame()
    colour = cv2.rotate(colour, cv2.ROTATE_90_COUNTERCLOCKWISE)
    with open(posefile, "rb") as f:
                pose_data = pickle.load(f)


    timestamp = params["timestamp"]
    shifted_timestamps_poses = [x - timestamp for x in pose_data["timestamps"]]
    nearest_timestamp_index = np.abs(shifted_timestamps_poses).argmin()

    # extract positions from pkl file dictionary
    left_shoulder_x = pose_data["locations"][nearest_timestamp_index][0][0]
    left_shoulder_y = pose_data["locations"][nearest_timestamp_index][0][1]
    right_shoulder_x = pose_data["locations"][nearest_timestamp_index][1][0]
    right_shoulder_y = pose_data["locations"][nearest_timestamp_index][1][1]
    left_hip_x = pose_data["locations"][nearest_timestamp_index][2][0]
    left_hip_y = pose_data["locations"][nearest_timestamp_index][2][1]
    right_hip_x = pose_data["locations"][nearest_timestamp_index][3][0]
    right_hip_y = pose_data["locations"][nearest_timestamp_index][3][1]

    # calculate ROI coordinates lying on back
    if left_hip_x > right_hip_x:
        rectangle_y_1 = int(min(left_shoulder_y, right_shoulder_y))
        rectangle_y_2 = int(max(left_hip_y, right_hip_y))
        rectangle_x_2 = int(max(left_hip_x, left_shoulder_x))
        rectangle_x_1 = int(min(right_shoulder_x, right_hip_x))
       
    else:
        rectangle_y_1 = int(min(left_shoulder_y, right_shoulder_y))
        rectangle_y_2 = int(max(left_hip_y, right_hip_y))
        rectangle_x_1 = int(min(left_hip_x, left_shoulder_x))
        rectangle_x_2 = int(max(right_shoulder_x, right_hip_x))


    colour2=colour.copy()
    roi_img=cv2.rectangle(colour2, (rectangle_x_1, rectangle_y_1), (rectangle_x_2, rectangle_y_2), (255,0,255), 5, 3)[:, :, [2, 1, 0]]

    ROI_x = rectangle_x_2 - rectangle_x_1
    ROI_y = rectangle_y_2 - rectangle_y_1
    ROI_size = ROI_x * ROI_y
    region_of_interest_data=np.zeros((ROI_size,3), dtype='i').reshape(ROI_y,ROI_x,3)
    for l in range(ROI_y):
        for m in range(ROI_x):
                region_of_interest_data[l][m]=colour[l + rectangle_y_1][m + rectangle_x_1]

    region_of_interest_data=region_of_interest_data[:, :, [2, 1, 0]]       
    mask_img=skin_pixel_identification(colour)

    mask_img_roi=np.zeros((ROI_size,3), dtype='i').reshape(ROI_y,ROI_x,3)
    for l in range(ROI_y):
        for m in range(ROI_x):
                mask_img_roi[l][m]=mask_img[l + rectangle_y_1][m + rectangle_x_1]      

    mask_img_roi=mask_img_roi[:, :, [2, 1, 0]]
    return roi_img, mask_img_roi, region_of_interest_data

########################################################################
params = {"ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "text.usetex" : False,
        "font.family" : "serif",
        "font.sans-serif": "Helvetica",
        }
plt.rcParams.update(params)

fig, axs = plt.subplots(3,6, figsize=(8.27, 6))

#501
file=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk501\file_20231005_153246.mkt"
poses=r"C:\Users\silas\Master Project\pose estimation\mk501\recording_20231005_144809_pose_corrected.pkl"
roi_img, mask_img_roi, region_of_interest_data=generate_plot_images(file, poses)


roi_img2=cv2.rectangle(roi_img.copy(), (300, 450), (420, 550), (0, 0, 0), -1)
roi_img2=cv2.rectangle(roi_img2, (270, 800), (310, 830), (0, 0, 0), -1)
axs[0,0].imshow(roi_img2)
#axs[0,0].set_axis_off()
axs[0,0].set_ylabel("(a)", fontsize=8, rotation=0, labelpad=10)
axs[0,0].set_title("(1)", fontsize=8)
axs[0, 0].spines["right"].set_visible(False)
axs[0, 0].spines["left"].set_visible(False)
axs[0, 0].spines["top"].set_visible(False)
axs[0, 0].spines["bottom"].set_visible(False)
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[1,0].imshow(region_of_interest_data)

#axs[1,0].set_axis_off()
axs[1,0].set_ylabel("(b)", fontsize=8, rotation=0,labelpad=10)
axs[1, 0].spines["right"].set_visible(False)
axs[1, 0].spines["left"].set_visible(False)
axs[1, 0].spines["top"].set_visible(False)
axs[1, 0].spines["bottom"].set_visible(False)
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])

     
axs[2,0].imshow(mask_img_roi)
#axs[2,0].set_axis_off()

axs[2,0].set_ylabel("(c)", fontsize=8, rotation=0, labelpad=10)
axs[2, 0].spines["right"].set_visible(False)
axs[2, 0].spines["left"].set_visible(False)
axs[2, 0].spines["top"].set_visible(False)
axs[2, 0].spines["bottom"].set_visible(False)
axs[2,0].set_xticks([])
axs[2,0].set_yticks([])


#502
file=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk502\file_20231009_162453.mkt"
poses=r"C:\Users\silas\Master Project\pose estimation\mk502\recording_20231009_162223_pose_corrected.pkl"

roi_img, mask_img_roi, region_of_interest_data=generate_plot_images(file, poses)
axs[0,1].set_title("(2)", fontsize=8)

axs[0,1].imshow(roi_img)
axs[0,1].set_axis_off()
             
axs[1,1].imshow(region_of_interest_data)
axs[1,1].set_axis_off()
     
axs[2,1].imshow(mask_img_roi)
axs[2,1].set_axis_off()



#503
file=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk503\file_20231026_082106.mkt"
poses=r"C:\Users\silas\Master Project\pose estimation\mk503\recording_20231026_081936_pose_corrected.pkl"
roi_img, mask_img_roi, region_of_interest_data=generate_plot_images(file, poses)

roi_img2=cv2.rectangle(roi_img.copy(), (350, 450), (490, 550), (0, 0, 0), -1)
axs[0,2].imshow(roi_img2)
axs[0,2].set_axis_off()
        
        
axs[0,2].set_title("(3)", fontsize=8)     
axs[1,2].imshow(region_of_interest_data)
axs[1,2].set_axis_off()
     
axs[2,2].imshow(mask_img_roi)
axs[2,2].set_axis_off()

#mk047
file=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk047\file_20230718_224807.mkt"
poses=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk047\part1_results_10frames_corrected.pkl"

roi_img, mask_img_roi, region_of_interest_data=generate_plot_images(file, poses)
#roi_img2=cv2.rectangle(roi_img.copy(), (170, 450), (340, 550), (0, 0, 0), -1)
axs[0,3].imshow(roi_img)
axs[0,3].set_axis_off()
axs[0,3].set_title("(4)", fontsize=8)          
axs[1,3].imshow(region_of_interest_data)
axs[1,3].set_axis_off()
     
axs[2,3].imshow(mask_img_roi)
axs[2,3].set_axis_off()


#mk026
file=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk026\file_20220711_175644.mkt"
poses=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk026\part1_results_10frames_corrected.pkl"

roi_img, mask_img_roi, region_of_interest_data=generate_plot_images(file, poses)
roi_img2=cv2.rectangle(roi_img.copy(), (270, 500), (450, 620), (0, 0, 0), -1)
axs[0,4].imshow(roi_img2)
axs[0,4].set_axis_off()
axs[0,4].set_title("(5)", fontsize=8)        
axs[1,4].imshow(region_of_interest_data)
axs[1,4].set_axis_off()
     
axs[2,4].imshow(mask_img_roi)
axs[2,4].set_axis_off()


#mk045
file=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk045\file_20230705_174314.mkt"
poses=r"C:\Users\silas\Master Project\meerkat raw data\example_files\mk045\part1_results_10frames_corrected.pkl"

roi_img, mask_img_roi, region_of_interest_data=generate_plot_images(file, poses)
roi_img2=cv2.rectangle(roi_img.copy(), (250, 450), (390, 550), (0, 0, 0), -1)
axs[0,5].imshow(roi_img2)
axs[0,5].set_axis_off()
axs[0,5].set_title("(6)", fontsize=8)         
axs[1,5].imshow(region_of_interest_data)
axs[1,5].set_axis_off()
     
axs[2,5].imshow(mask_img_roi)
axs[2,5].set_axis_off()



plt.tight_layout(pad=1)
plt.show()

