import os
import pywt
import cv2
import numpy as np
import torch
import torch.nn as nn

# 定义注意力机制模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        
    def forward(self, x):
        attention_scores = torch.sigmoid(self.weight)
        weighted_features = x * attention_scores.expand_as(x)
        output = torch.sum(weighted_features, dim=1, keepdim=True)
        return output

# 定义单个系数的融合函数
def fuseSingleCoeff(coef1, coef2, coef3, method, dynamic_weight1, dynamic_weight2, dynamic_weight3):
    if (method == 'mean'):
        coef = (coef1 + coef2 + coef3) / 3
    elif (method == 'min'):
        coef = np.minimum(coef1, np.minimum(coef2, coef3))
    elif (method == 'max'):
        coef = np.maximum(coef1, np.maximum(coef2, coef3))
    
    # 动态权重调整
    coef = coef * dynamic_weight1 * dynamic_weight2 * dynamic_weight3
    return coef

# 计算图像平均亮度
def calculate_average_brightness(image):
    return np.mean(image)

# 定义图像融合函数
def fuseImages(image_folder1, image_folder2, image_folder3, output_folder):
    for file_name in os.listdir(image_folder1):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            # 读取三张图像
            image1 = cv2.imread(os.path.join(image_folder1, file_name))
            image2 = cv2.imread(os.path.join(image_folder2, file_name))
            image3 = cv2.imread(os.path.join(image_folder3, file_name))

# 确保图像具有相同的分辨率
            width, height =224, 224  # 你可以根据实际情况选择一个合适的大小
            image1 = cv2.resize(image1, (width, height))
            image2 = cv2.resize(image2, (width, height))
            print("宽度:", width)
            print("高度:", height)
            image3 = cv2.resize(image3, (width, height))

            if image3 is not None and not image3.size == 0:
               image3 = cv2.resize(image3, (width, height))
    # 继续处理调整大小后的图像
            else:
                print("图像为空或大小为零。")



            # 确保图像具有相同的像素位宽
            if image1.dtype != image2.dtype or image1.dtype != image3.dtype:
                # 将图像像素位宽转换为一致
                target_dtype = np.max([image1.dtype, image2.dtype, image3.dtype])
                image1 = image1.astype(target_dtype)
                image2 = image2.astype(target_dtype)
                image3 = image3.astype(target_dtype)

            # 计算图像平均亮度
            brightness_image1 = calculate_average_brightness(image1)
            brightness_image2 = calculate_average_brightness(image2)
            brightness_image3 = calculate_average_brightness(image3)

            # 根据平均亮度调整权重
            base_weight_ir =dynamic_weight_image2= 0.8  # 基础红外图像权重
            brightness_threshold = 24  # 亮度阈值，可以根据实际情况调整

            if brightness_image1 > brightness_threshold:
                dynamic_weight_image1 = base_weight_ir
            else:
                dynamic_weight_image1 = base_weight_ir * 0.4
            if brightness_image3 > brightness_threshold:
                dynamic_weight_image3 = base_weight_ir
            else:
                dynamic_weight_image3 = base_weight_ir * 0.4

            # First: Do wavelet transform on each image
            wavelet = 'db2'
            cooef1 = pywt.wavedec2(image1[:, :], wavelet, level=1)
            cooef2 = pywt.wavedec2(image2[:, :], wavelet, level=1)
            cooef3 = pywt.wavedec2(image3[:, :], wavelet, level=1)
            # 使用动态权重进行系数融合
            fusedCooef = []
            for i in range(len(cooef1)):
                if i == 0:
                    fusedCooef.append(fuseSingleCoeff(cooef1[0], cooef2[0], cooef3[0], 'mean', dynamic_weight_image1, dynamic_weight_image2, dynamic_weight_image3))
                else:
                    c1 = fuseSingleCoeff(cooef1[i][0], cooef2[i][0], cooef3[i][0], 'mean', dynamic_weight_image1, dynamic_weight_image2, dynamic_weight_image3)
                    c2 = fuseSingleCoeff(cooef1[i][1], cooef2[i][1], cooef3[i][1], 'mean', dynamic_weight_image1, dynamic_weight_image2, dynamic_weight_image3)
                    c3 = fuseSingleCoeff(cooef1[i][2], cooef2[i][2], cooef3[i][2], 'mean', dynamic_weight_image1, dynamic_weight_image2, dynamic_weight_image3)
                    fusedCooef.append((c1, c2, c3))

            # Third: After we fused the coefficients we need to transform back to get the image
            fusedImage = pywt.waverec2(fusedCooef, wavelet)

            # Forth: Normalize values to be in uint8
            fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage), (np.max(fusedImage) - np.min(fusedImage))), 255)
            fusedImage = fusedImage.astype(np.uint8)

            # Save fused image to output folder
            output_path = os.path.join(output_folder, f"{file_name[:-4]}.jpg")
            cv2.imwrite(output_path, fusedImage)

# 指定图像文件夹和输出文件夹
image_folder1 = '/'
image_folder2 = '/'
image_folder3 = '/'
output_folder = '/'


# 调用 fuseImages 函数，传递 attention_module 参数
fuseImages(image_folder1, image_folder2, image_folder3, output_folder)

