import cv2
import os
import numpy as np

def cropimg(patch,step, imgpath,save_path):
	"patch: the cropped patch size, default:128; step: cropping gap, default:128"
	frame1 = cv2.imread(imgpath)
	W = frame1.shape[1]
	H = frame1.shape[0]
	for i in range(0, H - patch, step):
		for j in range(0, W - patch, step):
			img_nm = os.path.splitext(imgpath)[0]
			img_nmm = img_nm.split('/')[-1]
			cropImg1 = frame1[j:j + patch, i:i + patch]
			hist, bins = np.histogram(cropImg1.ravel(), 256, [0, 256])
			rate = (hist[0]/(patch*patch))*100
			if rate > 90:
				print('this patch is black!') #remove black patches
			else:
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				cv2.imwrite(save_path + '/' + img_nmm + '_' + str(i) + '_' + str(j) + '.jpg', cropImg1)
				print(save_path + '/' + img_nmm + '_' + str(i) + '_' + str(j) + '.jpg')

if __name__ == '__main__':
	source_path = "./train/gt/"
	save_path = "./train_dataset/gt"
	file_names = os.listdir(source_path)

	for i in range(len(file_names)):
		x = cropimg(128, 128, source_path + file_names[i], save_path)
		print("crop", file_names[i])
		print("crop_nums", i)

	print("cropping over!")
