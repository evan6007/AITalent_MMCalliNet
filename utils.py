import cv2
import numpy as np
from functools import cmp_to_key
import base64
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import io
import json

def remove_red(img_cv):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Define a wide range around the red color in HSV space, split into two parts
    lower_red1 = np.array([0, 30, 100])
    upper_red1 = np.array([30, 255, 255])

    lower_red2 = np.array([150, 30, 100])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

    # Combine the masks
    mask = mask1 | mask2

    # Replace all red (now white in the mask) pixels with white in the original image
    img_rgb[mask > 0] = [255, 255, 255]

    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    return img_rgb

def find_square_corners(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 進行二值化處理（根據圖像特性選擇適合的閾值方法）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # 使用形態學操作來擴展網格線（根據圖像特性選擇適合的核大小）
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)


    # Find contours in the edge map
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the corners
    squares = []

    # 遍歷輪廓，繪製矩形框
    min_area_threshold = 50000  # 設定最小面積閾值
    max_area_threshold = 3000000  # 設定最大面積閾值

    # Loop over the contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.045 * peri, True)

        # If our approximated contour has four points, we can assume that we have found a grid
        if len(approx) == 4:
            # Check if all angles are approximately 90 degrees
            x, y, w, h = cv2.boundingRect(c)
            area = w * h  # 計算矩形框的面積
            if area > min_area_threshold and area < max_area_threshold: #超過一定的面積
                cos = []
                for i in range(2, 5):
                    cos.append(angle(approx[i%4], approx[i-2], approx[i-1]))
                    cos.sort()
                    ratio = w / float(h)  # assuming h != 0
                    if 0.8 <= ratio <= 1.2:  # checks if w and h are within 90% of each other
                        squares.append(approx)
                    
    return squares

# Function to return needed angle
def angle(pt1, pt2, pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return (dx1*dx2 + dy1*dy2)/np.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2)) + 1e-10
                               
# 去除重疊的函式
def remove_overlapping_squares(squares, overlap_threshold=0.8):
    # Calculate the bounding rectangles of the squares
    rects = [cv2.boundingRect(square) for square in squares]

    # Initialize a list to hold the non-overlapping squares
    non_overlapping_squares = []

    # Loop over the squares
    for i in range(len(squares)):
        # Assume that the square does not overlap with any other square
        overlaps = False

        # Calculate the area of the square
        area1 = rects[i][2] * rects[i][3]

        # Loop over the other squares
        for j in range(i + 1, len(squares)):
            # Calculate the intersection of the two bounding rectangles
            x = max(rects[i][0], rects[j][0])
            y = max(rects[i][1], rects[j][1])
            w = min(rects[i][0] + rects[i][2], rects[j][0] + rects[j][2]) - x
            h = min(rects[i][1] + rects[i][3], rects[j][1] + rects[j][3]) - y

            # If there is an intersection, calculate the area of the intersection
            if w > 0 and h > 0:
                intersection_area = w * h

                # Calculate the area of the other square
                area2 = rects[j][2] * rects[j][3]

                # If the area of the intersection is greater than the threshold for either square, mark the square as overlapping
                if intersection_area > overlap_threshold * area1 or intersection_area > overlap_threshold * area2:
                    overlaps = True
                    break

        # If the square does not overlap with any other square, add it to the list of non-overlapping squares
        if not overlaps:
            non_overlapping_squares.append(squares[i])

    return non_overlapping_squares

def mask(image):
    # Create a copy of the original image
    output = image.copy()

    # Define the rectangle
    x1, y1, x2, y2 = 316, 838, 880, 1405

    # Create a black mask with the same dimensions as the image
    mask = np.zeros_like(image)

    # Set the region inside the rectangle to white
    mask[y1:y2, x1:x2] = [255, 255, 255]

    # Split the mask and the image into their respective channels
    mask_b, mask_g, mask_r = cv2.split(mask)
    output_b, output_g, output_r = cv2.split(output)

    # Use the mask to change all black (now white in the mask) pixels to white in the original image
    output_b[mask_b == 0] = 255
    output_g[mask_g == 0] = 255
    output_r[mask_r == 0] = 255

    # Merge the channels back into a single image
    output = cv2.merge([output_b, output_g, output_r])

    return output

# Function to draw squares on the image
def draw_squares(image, squares):
    for square in squares:
        cv2.drawContours(image, [square], -1, (0, 255, 0), 3)

def order_points(pts):
    # 重新塑形點，以便我們可以使用 numpy 的函數
    print("pts=",pts.shape)
    pts = pts.reshape((4,2))

    # 分別計算點的 x 和 y 坐標的總和和差
    s = pts.sum(axis=1)
    d = np.subtract(pts[:, 0], pts[:, 1])

    # 左上角的點具有最小的 x + y 值
    tl = pts[np.argmin(s)]
    # 右上角的點具有最大的 x - y 值
    tr = pts[np.argmax(d)]
    # 右下角的點具有最大的 x + y 值
    br = pts[np.argmax(s)]
    # 左下角的點具有最小的 x - y 值
    bl = pts[np.argmin(d)]

    return np.array([tl, tr, br, bl], dtype="float32")


# 定義比較函數
def compare_y(square1, square2):
    center1 = square1.mean(axis=0)
    center2 = square2.mean(axis=0)
    return -1 if center1[1] > center2[1] else 1

def compare_x(square1, square2):
    center1 = square1.mean(axis=0)
    center2 = square2.mean(axis=0)
    return -1 if center1[0] > center2[0] else 1


def path_cropimg_to_eval(path=None):
    imglist = []
    # Load the image
    image = cv2.imread(path)
    image = remove_red(image)
    #image = mask(image)
    squares = find_square_corners(image)
    squares = remove_overlapping_squares(squares)
    #先把squares裡面的四個點依序左上、右上、右下、左下排序
    squares = [order_points(square) for square in squares]
    # 對y座標排序，再對x座標排序 讓位置是右上開始往下數，一直數到左下。
    squares = sorted(squares, key=cmp_to_key(compare_y))
    squares = sorted(squares, key=cmp_to_key(compare_x))
    #warp
    src_pts = np.array([squares[0][0], squares[0][1], squares[0][2],squares[0][3]], dtype="float32")
    dst_pts = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype="float32") # we choose 300x400 as the size of the output image
    # Compute the perspective transformation matrix and then apply it
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (500, 500))
    for j in range(5):
        for k in range(5):
            imglist.append(warped[100*j:100*(j+1),100*k:100*(k+1)])
    x_data = np.array(imglist)[np.newaxis, ...]
    x_data = np.transpose(x_data, (0, 1, 4, 2, 3))

    return x_data


#UI
def find_coner_point(pts):
    # 分別計算點的 x 和 y 坐標的總和和差
    s = pts.sum(axis=1)
    d = np.subtract(pts[:, 0], pts[:, 1])
    # 左上角的點具有最小的 x + y 值
    tl = pts[np.argmin(s)]
    # 右上角的點具有最大的 x - y 值
    tr = pts[np.argmax(d)]
    # 右下角的點具有最大的 x + y 值
    br = pts[np.argmax(s)]
    # 左下角的點具有最小的 x - y 值
    bl = pts[np.argmin(d)]

    return  tl, tr, br, bl


def img_cropimg_to_eval(data=None):
    imglist = []
    # Load the image

    base64_img = data['imageData']
    img_data = base64.b64decode(base64_img)
    nparr = np.fromstring(img_data, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 这是原始图像
    display_width = data['displayWidth']  # 从前端传递的显示图像的宽度
    display_height = data['displayHeight']  # 从前端传递的显示图像的高度
    scale_factor_width = original_img.shape[1] / display_width
    scale_factor_height = original_img.shape[0] / display_height


    points = data['points']
    # 使用缩放因子将选定的点映射回原始图像坐标
    scaled_points = []
    for point in points:
        scaled_x = point['x'] * scale_factor_width
        scaled_y = point['y'] * scale_factor_height
        scaled_points.append((scaled_x, scaled_y))
    
    scaled_points = np.array(scaled_points, dtype=np.int32)
    tl, tr, br, bl = find_coner_point(scaled_points)
    original_img = remove_red(original_img)
    src_pts = np.array([tl, tr, br, bl], dtype="float32")
    dst_pts = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(original_img, matrix, (500, 500))
    #v2.imshow("warped",warped)
    #cv2.waitKey()
    for j in range(5):
        for k in range(5):
            imglist.append(warped[100*j:100*(j+1),100*k:100*(k+1)])
    x_data = np.array(imglist)[np.newaxis, ...]
    x_data = np.transpose(x_data, (0, 1, 4, 2, 3))

    return x_data


#畫雷達圖
def plot_radar_chart(values, color, alpha, ylim, fontsize=20):
    labels = ['FSIQ', 'VCI', 'WMI', 'PRI/VSI', 'PSI']
    label_colors = ['#ea4435a9', '#fbbd05a4', '#34a853b1', '#000406b6', '#4286f4ac']  # 這裡可以設定每個標籤的顏色
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles = [(a + np.pi / 2 - 72* 3 * (np.pi / 180)) % (2 * np.pi) for a in angles]
    
    fsiq_index = angles.index(min(angles))
    labels = [labels[fsiq_index]] + labels[fsiq_index+1:] + labels[:fsiq_index]
    values = [values[fsiq_index]] + values[fsiq_index+1:] + values[:fsiq_index]
    
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor("none")
    ax.fill(angles, values, color=color, alpha=alpha)
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.xaxis.set_tick_params(pad=20)
    
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(label_colors[i])
    
    
    for label, angle, value in zip(labels, angles, values):
        ax.text(angle, value, str(round(value, 2)), color='black', size=15, horizontalalignment='center', verticalalignment='top')
    ax.set_ylim(0, ylim)
    

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

#畫高斯圖
def plot_gaussian(values):
    mu = 100
    sigma = 15
    x = np.linspace(65, 135, 1000)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- 0.5 * ((x - mu) / sigma)**2)
    max_y = max(y)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Set the entire background to white
    ax.set_facecolor('white')
    ax.set_xlim(65, 135)
    ax.set_ylim(0, max_y)
    
    # Define colors for different standard deviations
    colors = ["#fa8888", "#fcabab", "#fcdcdc"]
    
    # Fill areas for different standard deviations
    ax.fill_between(x, y, where=((x < mu + sigma) & (x > mu - sigma)), color=colors[2])
    ax.fill_between(x, y, where=((x < mu + 2*sigma) & (x > mu + sigma)) | ((x < mu - sigma) & (x > mu - 2*sigma)), color=colors[1])
    ax.fill_between(x, y, where=((x < mu + 3*sigma) & (x > mu + 2*sigma)) | ((x < mu - 2*sigma) & (x > mu - 3*sigma)), color=colors[0])   
    
    # Plot the Gaussian distribution
    ax.plot(x, y, 'black', lw=1)
    
    labels = ['FSIQ', 'VCI', 'WMI', 'PRI/VSI', 'PSI']
    
    coordinates = []
    for label, val in zip(labels, values):
        disp_coord = ax.transData.transform_point((val, 0.01))
        coordinates.append((label, val, disp_coord))
    converted_data = [(item[0],item[1], list(item[2])) for item in coordinates]
    
    ax.set_title('Gaussian Distribution with Corrected Shades')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return image_base64,json.dumps(converted_data)


