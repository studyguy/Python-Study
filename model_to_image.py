# coding: utf-8
import os
import sys
import copy
import struct
import argparse
import numpy as np
import math
import cv2
import json
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm

def read_geo(file_path_name):
    fo = open(file_path_name, "r")
    geo_dict = {}
    for line in fo.readlines(): # 依次读取每行  
        line = line.strip() # 去掉每行头尾空白  
        elements = line.split(' ')
        if len(elements) == 4: # 标准geos
            geo_dict[elements[0]] = [float(elements[1]), float(elements[2]), float(elements[3])]
        elif len(elements) == 5: # 带有broken match数量的geos
            geo_dict[elements[0]] = [float(elements[1]), float(elements[2]), float(elements[3]), float(elements[4])]
        else:
            assert False, 'geos file has wrong format!'
    fo.close()
    return geo_dict

def value_to_color(color_map_rgb, color_map_resolution, min_value, max_value, cur_value):
    '''
    color_map_rgb是函数外生成的，只需要生成一次，可以用如下代码生成：
    color_map_resolution = 100
    color_map_x = np.linspace(0.0, 1.0, color_map_resolution+1)
    color_map_rgb = cm.get_cmap('jet')(color_map_x)[np.newaxis, :, :3]
    '''
    if cur_value < min_value:
        cur_value = min_value
    if cur_value > max_value:
        cur_value = max_value
    cur_value = (max_value - cur_value) / (max_value - min_value)
    cur_value = int(cur_value * color_map_resolution)
    cur_rgb = color_map_rgb[0][cur_value]
    cur_rgb = cur_rgb * 255
    cur_rgb_tuple = (int(cur_rgb[0]), int(cur_rgb[1]), int(cur_rgb[2]))
    return cur_rgb_tuple

def valid_grid_num_to_gray_image(XY_image, XY_RowCol, top_sup_percent):
    validnum = np.sum(XY_image > 0) # 非0元素的个数 
    validpercent =  100 - validnum * 1.0 / XY_image.size * top_sup_percent #压缩前1%的点云密集区域

    maxthr = np.percentile(XY_image,validpercent)
    print("validnum/totalgrid", validnum, "/", XY_image.size, " validpercent", validpercent, " maxthr", maxthr)

    XY_image[XY_image > maxthr] = maxthr # 统一上边界
    XY_image = XY_image / maxthr * 255.0 
    XY_image = 255 - XY_image.astype(np.uint8)

    # matrix变成彩色图
    tmp = np.zeros((XY_image.shape[0], XY_image.shape[1], 3), np.uint8)
    for idx in range(3):
        tmp[:,:,idx] = XY_image
    XY_image = tmp
    return XY_image

# 路径解析，单个路径，或[path1,path2]格式的多个路径
def path_parser(path):
    if path[0] == '[' and path[-1] ==']':
        pathstr = path[1:-1]
        path_list = pathstr.split(',')
    else:
        path_list = [path]
    return path_list    

def convert_ply_to_image(ply_path, image_save_path, geos_path, resolution, top_sup_percent = 1):
    '''
    //      ___________________________________Y -> pixel_x(col)
    //      |             ^ X                 |
    //      |             |                   |
    //      |             |                   |
    //      |             |                   |
    //      | Y<-----------Z(out)             |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |_________________________________|
    //      X -> pixel_y(row)

    //      ___________________________________X -> pixel_x(col)
    //      |             ^ Z                 |
    //      |             |                   |
    //      |             |                   |
    //      |             |                   |
    //      | X<-----------Y(out)             |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |_________________________________|
    //      Z -> pixel_y(row)

    //      ___________________________________Y -> pixel_x(col)
    //      |             ^ Z                 |
    //      |             |                   |
    //      |             |                   |
    //      |             |                   |
    //      | Y<-----------X(in)              |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |                                 |
    //      |_________________________________|
    //      Z -> pixel_y(row)
    '''
    ply_path_list = path_parser(ply_path)
    (image_save_path, filename) = os.path.split(image_save_path)
    image_save_path = image_save_path + "/"

    X_max = -sys.maxsize
    Y_max = -sys.maxsize
    Z_max = -sys.maxsize
    X_min = sys.maxsize
    Y_min = sys.maxsize
    Z_min = sys.maxsize

    plydatas = {}
  #   for (key,value) in a.items()
    for ply_file in ply_path_list:    
        # ply数据结构转换成np.array
        plydata = PlyData.read(ply_file)
        plydatas[ply_file] = plydata

        X = plydata['vertex']['x']
        Y = plydata['vertex']['y']
        Z = plydata['vertex']['z']

        xarr = np.array([X])
        yarr = np.array([Y])
        zarr = np.array([Z])
        percent = 0.5
        X_max = max(X_max, np.percentile(xarr, 100-percent))
        X_min = min(X_min, np.percentile(xarr, percent))
        Y_max = max(Y_max, np.percentile(yarr, 100-percent))
        Y_min = min(Y_min, np.percentile(yarr, percent))
        Z_max = max(Z_max, np.percentile(zarr, 100-percent))
        Z_min = min(Z_min, np.percentile(zarr, percent))

    # 点云边界
    # X_min = min(X)
    # Y_min = min(Y)
    # Z_min = min(Z)
    # # X_max = max(X)
    # # Y_max = max(Y)
    # Z_max = max(Z)

    geo_dict = None
    if os.path.isfile(geos_path):
        # 如果存在geo，选择轨迹和点云中更大的边界绘制图像
        geo_dict = read_geo(geos_path)
        assert len(geo_dict) > 0, 'Error! empty geos!'
        for _, value in geo_dict.items():
            X_min = min(value[0], X_min)
            Y_min = min(value[1], Y_min)
            Z_min = min(value[2], Z_min)
            X_max = max(value[0], X_max)
            Y_max = max(value[1], Y_max)
            Z_max = max(value[2], Z_max)
    # 为了好选择轨迹，往外扩点距离
    boundary_out_length = 5.0
    X_min -= boundary_out_length
    Y_min -= boundary_out_length
    Z_min -= boundary_out_length
    X_max += boundary_out_length
    Y_max += boundary_out_length
    Z_max += boundary_out_length

    # 构建图像网格
    resolution_inv = 1.0 / resolution
    X_max_idx = int(math.ceil((X_max * resolution_inv)))
    Y_max_idx = int(math.ceil((Y_max * resolution_inv)))
    Z_max_idx = int(math.ceil((Z_max * resolution_inv)))

    X_min_idx = int(math.floor((X_min * resolution_inv)))
    Y_min_idx = int(math.floor((Y_min * resolution_inv)))
    Z_min_idx = int(math.floor((Z_min * resolution_inv)))

    X_delta_idx = X_max_idx - X_min_idx
    Y_delta_idx = Y_max_idx - Y_min_idx
    Z_delta_idx = Z_max_idx - Z_min_idx


    XY_image_merged = np.zeros((X_delta_idx, Y_delta_idx))
    ZX_image_merged = np.zeros((Z_delta_idx, X_delta_idx))
    ZY_image_merged = np.zeros((Z_delta_idx, Y_delta_idx))
    # ZY_image = np.zeros((Z_delta_idx, Y_delta_idx, 3), np.uint8)
    
    for (ply_file,plydata) in plydatas.items():
    #for ply_file in ply_path_list: 
     #   plydata = PlyData.read(ply_file)
        X = plydata['vertex']['x']
        Y = plydata['vertex']['y']
        Z = plydata['vertex']['z']
        XY_image = np.zeros((X_delta_idx, Y_delta_idx))
        ZX_image = np.zeros((Z_delta_idx, X_delta_idx))
        ZY_image = np.zeros((Z_delta_idx, Y_delta_idx))
        XY = np.zeros((len(plydata['vertex']['x']), 2))
        ZX = np.zeros((len(plydata['vertex']['x']), 2))
        ZY = np.zeros((len(plydata['vertex']['x']), 2))
        XY[:, 0] = np.array(X)
        XY[:, 1] = np.array(Y)

        ZX[:, 0] = np.array(Z)
        ZX[:, 1] = np.array(X)

        ZY[:, 0] = np.array(Z)
        ZY[:, 1] = np.array(Y)

        # 统计每个网格落入3d点的个数
        XY_RowCol = [X_max_idx, Y_max_idx] - (np.ceil(XY * resolution_inv)).astype('int')
        for idx in range(XY_RowCol.shape[0]):
            if XY_RowCol[idx, 0] < 0 or XY_RowCol[idx, 0] >= X_delta_idx or XY_RowCol[idx, 1] < 0 or XY_RowCol[idx, 1] >= Y_delta_idx: 
                continue
            XY_image[XY_RowCol[idx, 0], XY_RowCol[idx, 1]] += 1
            XY_image_merged[XY_RowCol[idx, 0], XY_RowCol[idx, 1]] += 1

        ZX_RowCol = [Z_max_idx, X_max_idx] - (np.ceil(ZX * resolution_inv)).astype('int')
        for idx in range(XY_RowCol.shape[0]):
            if ZX_RowCol[idx, 0] < 0 or ZX_RowCol[idx, 0] >= Z_delta_idx or ZX_RowCol[idx, 1] < 0 or ZX_RowCol[idx, 1] >= X_delta_idx: 
                continue
            ZX_image[ZX_RowCol[idx, 0], ZX_RowCol[idx, 1]] += 1
            ZX_image_merged[ZX_RowCol[idx, 0], ZX_RowCol[idx, 1]] += 1

        ZY_RowCol = [Z_max_idx, Y_max_idx] - (np.ceil(ZY * resolution_inv)).astype('int')
        for idx in range(XY_RowCol.shape[0]):
            if ZX_RowCol[idx, 0] < 0 or ZX_RowCol[idx, 0] >= Z_delta_idx or ZY_RowCol[idx, 1] < 0 or ZY_RowCol[idx, 1] >= Y_delta_idx: 
                continue
            ZY_image[ZY_RowCol[idx, 0], ZY_RowCol[idx, 1]] += 1
            ZY_image_merged[ZY_RowCol[idx, 0], ZY_RowCol[idx, 1]] += 1

        XY_image = valid_grid_num_to_gray_image(XY_image, XY_RowCol, top_sup_percent)
        ZX_image = valid_grid_num_to_gray_image(ZX_image, ZX_RowCol, top_sup_percent)
        ZY_image = valid_grid_num_to_gray_image(ZY_image, ZY_RowCol, top_sup_percent)

        # 如果提供geos.txt, 画轨迹
        if os.path.isfile(geos_path):
            # 画geos.txt
            # geo_dict = read_geo(geos_path)
            # assert len(geo_dict) > 0, 'Error! empty geos!'
            geo_broken_match = None
            for key, value in geo_dict.items():
                if len(geo_dict[key]) == 4:
                    geo_broken_match = np.zeros((len(geo_dict), 1))
                break
            geo_XY = np.zeros((len(geo_dict), 2))
            geo_ZY = np.zeros((len(geo_dict), 2))
            geo_ZX = np.zeros((len(geo_dict), 2))

            idx = 0
            for _, value in geo_dict.items():
                geo_XY[idx, 0] = value[0]
                geo_XY[idx, 1] = value[1]

                geo_ZX[idx, 0] = value[2]
                geo_ZX[idx, 1] = value[0]

                geo_ZY[idx, 0] = value[2]
                geo_ZY[idx, 1] = value[1]

                if geo_broken_match is not None:
                    geo_broken_match[idx] = value[3]

                idx += 1
            
            geo_XY_traj = [X_max_idx, Y_max_idx] - (np.ceil(geo_XY * resolution_inv)).astype('int')
            geo_ZX_traj = [Z_max_idx, X_max_idx] - (np.ceil(geo_ZX * resolution_inv)).astype('int')
            geo_ZY_traj = [Z_max_idx, Y_max_idx] - (np.ceil(geo_ZY * resolution_inv)).astype('int')
            # 提前生成color map
            # 这个速度比较慢，且只需要运行一次
            color_map_resolution = 100
            color_map_x = np.linspace(0.0, 1.0, color_map_resolution+1)
            color_map_rgb = cm.get_cmap('jet')(color_map_x)[np.newaxis, :, :3]
            if geo_broken_match is not None:
                for idx in range(geo_XY.shape[0]):
                    cur_rgb_tuple = value_to_color(color_map_rgb, color_map_resolution, 0, 50, geo_broken_match[idx])
                    XY_image = cv2.circle(XY_image, (geo_XY_traj[idx, 1], geo_XY_traj[idx, 0]), 3, cur_rgb_tuple, 1)
                    ZX_image = cv2.circle(ZX_image, (geo_ZX_traj[idx, 1], geo_ZX_traj[idx, 0]), 3, cur_rgb_tuple, 1)
                    ZY_image = cv2.circle(ZY_image, (geo_ZY_traj[idx, 1], geo_ZY_traj[idx, 0]), 3, cur_rgb_tuple, 1)
            else:
                for idx in range(geo_XY.shape[0]):
                    XY_image = cv2.circle(XY_image, (geo_XY_traj[idx, 1], geo_XY_traj[idx, 0]), 3, (0, 0, 255), 1)
                    ZX_image = cv2.circle(ZX_image, (geo_ZX_traj[idx, 1], geo_ZX_traj[idx, 0]), 3, (0, 0, 255), 1)
                    ZY_image = cv2.circle(ZY_image, (geo_ZY_traj[idx, 1], geo_ZY_traj[idx, 0]), 3, (0, 0, 255), 1)
        
        # 画右手坐标系，最后画，不会被轨迹遮住
        draw_axis = True
        if (draw_axis):
            length_axis = 5.0
            axis_idx = int(math.ceil(length_axis * resolution_inv))

            XY_origin = [X_max_idx, Y_max_idx]
            XY_x_axis = [X_max_idx - axis_idx, Y_max_idx]
            XY_y_axis = [X_max_idx, Y_max_idx - axis_idx]
            XY_image = cv2.arrowedLine(XY_image, (XY_origin[1], XY_origin[0]), (XY_x_axis[1], XY_x_axis[0]), (0, 0, 255), 2) 
            XY_image = cv2.arrowedLine(XY_image, (XY_origin[1], XY_origin[0]), (XY_y_axis[1], XY_y_axis[0]), (0, 255, 0), 2)
        #    XY_image_merged = cv2.arrowedLine(XY_image_merged, (XY_origin[1], XY_origin[0]), (XY_x_axis[1], XY_x_axis[0]), (0, 0, 255), 2) 
        #    XY_image_merged = cv2.arrowedLine(XY_image_merged, (XY_origin[1], XY_origin[0]), (XY_y_axis[1], XY_y_axis[0]), (0, 255, 0), 2)

            ZX_origin = [Z_max_idx, X_max_idx]
            ZX_z_axis = [Z_max_idx - axis_idx, X_max_idx]
            ZX_x_axis = [Z_max_idx, X_max_idx - axis_idx]
            ZX_image = cv2.arrowedLine(ZX_image, (ZX_origin[1], ZX_origin[0]), (ZX_z_axis[1], ZX_z_axis[0]), (255, 0, 0), 2) 
            ZX_image = cv2.arrowedLine(ZX_image, (ZX_origin[1], ZX_origin[0]), (ZX_x_axis[1], ZX_x_axis[0]), (0, 0, 255), 2)
       #     ZX_image_merged = cv2.arrowedLine(ZX_image_merged, (ZX_origin[1], ZX_origin[0]), (ZX_z_axis[1], ZX_z_axis[0]), (255, 0, 0), 2) 
        #    ZX_image_merged = cv2.arrowedLine(ZX_image_merged, (ZX_origin[1], ZX_origin[0]), (ZX_x_axis[1], ZX_x_axis[0]), (0, 0, 255), 2)

            ZY_origin = [Z_max_idx, Y_max_idx]
            ZY_z_axis = [Z_max_idx - axis_idx, Y_max_idx]
            ZY_y_axis = [Z_max_idx, Y_max_idx - axis_idx]
            ZY_image = cv2.arrowedLine(ZY_image, (ZY_origin[1], ZY_origin[0]), (ZY_z_axis[1], ZY_z_axis[0]), (255, 0, 0), 2) 
            ZY_image = cv2.arrowedLine(ZY_image, (ZY_origin[1], ZY_origin[0]), (ZY_y_axis[1], ZY_y_axis[0]), (0, 255, 0), 2)
        #    ZY_image_merged = cv2.arrowedLine(ZY_image_merged, (ZY_origin[1], ZY_origin[0]), (ZY_z_axis[1], ZY_z_axis[0]), (255, 0, 0), 2) 
        #    ZY_image_merged = cv2.arrowedLine(ZY_image_merged, (ZY_origin[1], ZY_origin[0]), (ZY_y_axis[1], ZY_y_axis[0]), (0, 255, 0), 2)
        
        # 保存图像
        (fullpath_tmp, filename) = os.path.split(ply_file)
        
        XY_image_save_path = image_save_path + filename[0:-4] + '_XY.png'
        ZX_image_save_path = image_save_path + filename[0:-4] + '_ZX.png'
        ZY_image_save_path = image_save_path + filename[0:-4] + '_ZY.png'

        cv2.imwrite(XY_image_save_path, XY_image)
        cv2.imwrite(ZX_image_save_path, ZX_image)
        cv2.imwrite(ZY_image_save_path, ZY_image)

        # 保存配置文件, 2d->3d的转换方法
        XY_txt_save_path = image_save_path + filename[0:-4] + '_XY.txt'
        ZX_txt_save_path = image_save_path + filename[0:-4] + '_ZX.txt'
        ZY_txt_save_path = image_save_path + filename[0:-4] + '_ZY.txt'

        XY_txt = open(XY_txt_save_path, 'w')
        XY_txt.write('# XY_TRANSFORM->imagematrix_to_world:\n')
        XY_txt.write('# X = resolution * (max_X_ROW - image_row)\n')
        XY_txt.write('# Y = resolution * (max_Y_COL - image_col)\n')
        XY_txt.write('resolution(m):' + str(resolution) + '\n')
        XY_txt.write('max_X_ROW:' + str(X_max_idx) + '\n')
        XY_txt.write('max_Y_COL:' + str(Y_max_idx) + '\n')
        XY_txt.close()

        # 适配mapLT需要的json
        XY_json_save_path = image_save_path[0:-4] + '_XY.json'
        XY_json_data = {}
        XY_json_data['X'] = 'resolution * (max_X_ROW - image_row)'
        XY_json_data['Y'] = 'resolution * (max_Y_COL - image_col)'
        XY_json_data['unit'] = 'm'
        XY_json_data['resolution'] = str(resolution)
        XY_json_data['max_X_ROW'] = str(X_max_idx)
        XY_json_data['max_Y_COL'] = str(Y_max_idx)
        XY_json_data['T_visual_to_xx'] = np.identity(4, dtype=float).tolist()
        json_file = open(XY_json_save_path, 'w')
        json.dump(XY_json_data, json_file, indent=4)

        ZX_txt = open(ZX_txt_save_path, 'w')
        ZX_txt.write('# ZX_TRANSFORM->imagematrix_to_world:\n')
        ZX_txt.write('# Z = resolution * (max_Z_ROW - image_row)\n')
        ZX_txt.write('# X = resolution * (max_X_COL - image_col)\n')
        ZX_txt.write('resolution(m):' + str(resolution) + '\n')
        ZX_txt.write('max_Z_ROW:' + str(Z_max_idx) + '\n')
        ZX_txt.write('max_X_COL:' + str(X_max_idx) + '\n')
        ZX_txt.close()

        ZY_txt = open(ZY_txt_save_path, 'w')
        ZY_txt.write('# ZY_TRANSFORM->imagematrix_to_world:\n')
        ZY_txt.write('# Z = resolution * (max_Z_ROW - image_row)\n')
        ZY_txt.write('# Y = resolution * (max_Y_COL - image_col)\n')
        ZY_txt.write('resolution(m):' + str(resolution) + '\n')
        ZY_txt.write('max_Z_ROW:' + str(Z_max_idx) + '\n')
        ZY_txt.write('max_Y_COL:' + str(Y_max_idx) + '\n')
        ZY_txt.close()

    if len(ply_path_list) > 1:

        XY_image_merged = valid_grid_num_to_gray_image(XY_image_merged, XY_RowCol, top_sup_percent)
        ZX_image_merged = valid_grid_num_to_gray_image(ZX_image_merged, ZX_RowCol, top_sup_percent)
        ZY_image_merged = valid_grid_num_to_gray_image(ZY_image_merged, ZY_RowCol, top_sup_percent)

        draw_axis = True
        if (draw_axis):
            length_axis = 5.0
            axis_idx = int(math.ceil(length_axis * resolution_inv))

            XY_origin = [X_max_idx, Y_max_idx]
            XY_x_axis = [X_max_idx - axis_idx, Y_max_idx]
            XY_y_axis = [X_max_idx, Y_max_idx - axis_idx]
            XY_image_merged = cv2.arrowedLine(XY_image_merged, (XY_origin[1], XY_origin[0]), (XY_x_axis[1], XY_x_axis[0]), (0, 0, 255), 2) 
            XY_image_merged = cv2.arrowedLine(XY_image_merged, (XY_origin[1], XY_origin[0]), (XY_y_axis[1], XY_y_axis[0]), (0, 255, 0), 2)

            ZX_origin = [Z_max_idx, X_max_idx]
            ZX_z_axis = [Z_max_idx - axis_idx, X_max_idx]
            ZX_x_axis = [Z_max_idx, X_max_idx - axis_idx]
            ZX_image_merged = cv2.arrowedLine(ZX_image_merged, (ZX_origin[1], ZX_origin[0]), (ZX_z_axis[1], ZX_z_axis[0]), (255, 0, 0), 2) 
            ZX_image_merged = cv2.arrowedLine(ZX_image_merged, (ZX_origin[1], ZX_origin[0]), (ZX_x_axis[1], ZX_x_axis[0]), (0, 0, 255), 2)

            ZY_origin = [Z_max_idx, Y_max_idx]
            ZY_z_axis = [Z_max_idx - axis_idx, Y_max_idx]
            ZY_y_axis = [Z_max_idx, Y_max_idx - axis_idx]
            ZY_image_merged = cv2.arrowedLine(ZY_image_merged, (ZY_origin[1], ZY_origin[0]), (ZY_z_axis[1], ZY_z_axis[0]), (255, 0, 0), 2) 
            ZY_image_merged = cv2.arrowedLine(ZY_image_merged, (ZY_origin[1], ZY_origin[0]), (ZY_y_axis[1], ZY_y_axis[0]), (0, 255, 0), 2)
        

        XY_image_save_path = image_save_path + "_merged" + '_XY.png'
        ZX_image_save_path = image_save_path + "_merged" + '_ZX.png'
        ZY_image_save_path = image_save_path + "_merged" + '_ZY.png'

        cv2.imwrite(XY_image_save_path, XY_image_merged)
        cv2.imwrite(ZX_image_save_path, ZX_image_merged)
        cv2.imwrite(ZY_image_save_path, ZY_image_merged)

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', required=True)
    parser.add_argument('--geos_path', default='', type=str, help='geos.txt, 相机轨迹文件，有则绘制')
    parser.add_argument('--image_save_path' , required=True)
    parser.add_argument('--resolution', default=0.10, type=float, help='点云图分辨率，每像素=N米')
    parser.add_argument('--top_sup_percent', default=15, type=int, help="墨迹浓度，仅绘制点数占比前N%%的网格")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    convert_ply_to_image(args.ply_path, args.image_save_path, args.geos_path, args.resolution, args.top_sup_percent)
    
if __name__ == "__main__":
    main()