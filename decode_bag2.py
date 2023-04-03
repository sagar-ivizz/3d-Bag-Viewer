from turtle import color
import numpy as np
import rosbag
import cv2
import open3d as o3d
from PIL import Image
import struct



def get_pointcloud(color_image,depth_image,camera_intrinsics):

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                  np.linspace(0,image_height-1,image_height))
    camera_points_x = np.multiply(pixel_x-camera_intrinsics[0][2],depth_image/camera_intrinsics[0][0])
    camera_points_y = np.multiply(pixel_y-camera_intrinsics[1][2],depth_image/camera_intrinsics[1][1])
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]

    return camera_points,color_points

def write_pointcloud(filename,xyz_points,rgb_points=None):


    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffsss",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()

print('Reading data...')
bag = rosbag.Bag('new.bag')
c=[]
d=[]
print('Reordering data....')
count=0
for i in range(180):
        x='fc_'+str(i)
        for topic, msg, t in bag.read_messages(topics=[x]):
                c.append(msg)
        x='fd_'+str(i)
        for topic, msg, t in bag.read_messages(topics=[x]):
                d.append(msg)

vis = o3d.visualization.Visualizer()
vis.create_window( window_name='Demo',width=1280,height=720)

# geometry is the point cloud used in your animaiton
geometry = o3d.geometry.PointCloud()
param = o3d.io.read_pinhole_camera_parameters("viewpoint.json")
ctr = vis.get_view_control()




for i in range(180):
        print(i)
        K=[[441.9793701171875, 0.0, 425.7491760253906], [0.0, 441.9793701171875, 242.20574951171875], [0.0, 0.0, 1.0]]
        K1=[[457.4375, 0.0, 311.484375], [0.0, 457.078125, 253.353515625], [0.0, 0.0, 1.0]]
        rgb=np.reshape(c[i].data,(480,640,3))
        m = np.reshape(d[i].data, (480, 640))
        camera_points,color_points=get_pointcloud(rgb,m,K1)
        geometry.points = o3d.utility.Vector3dVector(camera_points)
        geometry.colors = o3d.utility.Vector3dVector(color_points/255)

        if i==4:
            normed = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            im = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
            cv2.imshow("image", im)
            cv2.waitKey(0)

        
        if i==0:
            vis.add_geometry(geometry)
            
        else:
            vis.update_geometry(geometry)
        
        #ctr.convert_from_pinhole_camera_parameters(param)
        vis.update_renderer()
        vis.poll_events()
        

        
vis.destroy_window()                
bag.close()

