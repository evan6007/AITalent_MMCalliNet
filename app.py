from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
import torch
from flask_cors import CORS
import torch.nn as nn
import torch.nn.functional as F

import subprocess
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io

from utils import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.maxpool3(self.relu3(self.conv3(x)))
        return x

class ModifiedMultiConvNet(nn.Module):
    def __init__(self, num_modules=25):
        super(ModifiedMultiConvNet, self).__init__()
        
        # Create a list of ConvModules
        self.modules_list = nn.ModuleList([CNN() for _ in range(num_modules)])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(25*12*12*128, 5)
        
    def forward(self, inputs):
        assert len(inputs) == len(self.modules_list), "Number of inputs should match number of modules"
        
        outputs = []
        for i, module in enumerate(self.modules_list):
            outputs.append(module(inputs[i]))

        concatenated_output = torch.cat(outputs, dim=1)  # Concatenate along the channel dimension
        flattened_output = self.flatten(concatenated_output)  
        final_output = self.fc(flattened_output)  # Pass through the FC layer

        return final_output


# Create an instance of the modified model with 25 modules
model = ModifiedMultiConvNet().to('cuda')
model.load_state_dict(torch.load('./weight/4_1_far4325model.pt'))
model.eval()


app = Flask(__name__)
CORS(app)



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle the uploaded file here
        pass

    return render_template('index.html')


@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    print("Received image data!")

    data = request.json
    print(type(data))

    if 'imageData1' in data:
        base64_img1 = data['imageData1']
        img_data = base64.b64decode(base64_img1)
        nparr = np.fromstring(img_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 这是原始图像
        display_width = data['displayWidth']  # 从前端传递的显示图像的宽度
        display_height = data['displayHeight']  # 从前端传递的显示图像的高度
        scale_factor_width = original_img.shape[1] / display_width
        scale_factor_height = original_img.shape[0] / display_height

        # Load the image
        image = original_img
        image = remove_red(image)
        squares = find_square_corners(image)
        squares = remove_overlapping_squares(squares)
        #先把squares裡面的四個點依序左上、右上、右下、左下排序
        squares = [order_points(square) for square in squares]
        # 對y座標排序，再對x座標排序 讓位置是右上開始往下數，一直數到左下。
        squares = sorted(squares, key=cmp_to_key(compare_y))
        squares = sorted(squares, key=cmp_to_key(compare_x))

        if squares and len(squares) > 0:
            squares[0][:, 0] /= scale_factor_width
            squares[0][:, 1] /= scale_factor_height
            squares[0] = np.round(squares[0]).astype(int)
            squares_list = squares[0].tolist()
            print("new squares=",squares_list)

            # 返回处理后的消息和图像
            return jsonify({"message": "Processed","squares": squares_list})
        else:
            return jsonify({"message": "Processed"})


    if 'imageData' in data:
        train_x_tensor = img_cropimg_to_eval(data)
        train_x_tensor = torch.Tensor(train_x_tensor)
        x = train_x_tensor.to("cuda")

        with torch.no_grad():
            pre_y = model([x[:,i] for i in range(25)])
            numpy_image = pre_y.cpu().squeeze().numpy()
            values = numpy_image.tolist()
            values = [round(num, 2) for num in values]
            normalized_values = [val / max(values) for val in values]
            
        
        image_base64_radar = plot_radar_chart(values, 'red', 0.25, 130)
        nor_image_base64_radar = plot_radar_chart(normalized_values, 'green', 0.25, 1)
            

        # 繪製高斯分布圖
        image_base64_gaussian,converted_data = plot_gaussian(values)
        print("converted_data: ",type(converted_data))

        return jsonify({
            "message": "Processed",
            "radar_image": image_base64_radar,
            "nor_radar_image": nor_image_base64_radar,
            "gaussian_image": image_base64_gaussian,
            "converted_data": converted_data,
            "FSIQ": round(float(numpy_image[0]),2),
            "VCI": round(float(numpy_image[1]),2),
            "WMI": round(float(numpy_image[2]),2),
            "PRI/VSI": round(float(numpy_image[3]),2),
            "PSI": round(float(numpy_image[4]),2)
        })
    else:
        return jsonify({"message":"error"})

        

if __name__ == '__main__':
    app.run(debug=True)
