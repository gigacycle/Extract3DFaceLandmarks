import json
import shutil
import time

import cv2
import face_alignment
import face_alignment.detection.sfd as face_detector_module
import torch
from vedo import *

import FaceFeatures as faceFeatures

# if you want to change the dataset path please change 'esrc_dataset_path'
esrc_dataset_path = './dataset/'


def calculate_2d_landmarks(img_dir, out_dir):
    if torch.cuda.is_available():
        dvc = 'cuda'
    else:
        dvc = 'cpu'
    try:
        # I had a problem with 'cuda', so I changed it to 'cpu'. You can try 'cuda' or dvc if you want.
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=dvc, flip_input=False)
        face_detector = face_detector_module.FaceDetector(device=dvc, verbose=False)
        print('Load images and corresponding landmarks ...')
        _, _, key_points = faceFeatures.preprocess(img_dir, out_dir, fa, face_detector, True, dvc)
        print('Processing landmarks ended')
        return key_points
    except Exception as e:
        print(f"Error on extracting 2D landmarks of : {img_dir} : {e}")
    return None


def calculate_3d_landmarks(obj, k_points_path, plotter, camera_settings_file):
    k_points_array = []
    w = 1000
    h = 1000

    if camera_settings_file:
        with open(camera_settings_file, "r") as json_file:
            camera_settings = json.load(json_file)
            plotter.camera.SetPosition(camera_settings["Position"])
            plotter.camera.SetFocalPoint(camera_settings["FocalPoint"])
            plotter.camera.SetViewUp(camera_settings["ViewUp"])
            w = int(camera_settings["Width"])
            h = int(camera_settings["Height"])

    with open(k_points_path, "r") as file:
        for line in file:
            x, y = map(float, line.strip().split(","))
            k_points_array.append((x, h - y))

    landmarks_3d = []
    for k_point in k_points_array:
        world_coord = plotter.at(0).compute_world_coordinate(objs=[obj], pos2d=k_point)
        if world_coord is not None:
            vertex_pos = plotter.at(0).get_meshes()[0].closest_point(world_coord)
            if vertex_pos is not None:
                landmarks_3d.append(vertex_pos)
    return landmarks_3d


def extract_esrc_landmarks():
    cnt = 0
    _plotter = None
    for sub_dir in os.listdir(esrc_dataset_path):
        cnt += 1
        # if sub_dir.lower().strip() not in objList:
        #     continue
        sub_dir_path = os.path.join(esrc_dataset_path, sub_dir)
        # if os.path.exists(os.path.join(sub_dir_path, object_path, '3D Landmarks', f"{sub_dir}_3D_landmarks.txt")):
        #     continue
        if os.path.isdir(sub_dir_path):
            object_path = os.path.join(sub_dir_path, f"{sub_dir}.obj")
            extract_landmarks(object_path, True)


def extract_landmarks(object_path, remove_extra_files):
    if not object_path.lower().endswith('.obj'):
        print('wrong 3d object file : ' + object_path)
        return
    sub_dir = os.path.splitext(os.path.basename(object_path))[0]
    sub_dir_path = os.path.dirname(object_path)
    _txt_path = os.path.join(sub_dir_path, f"{sub_dir}.jpg")
    _plotter = None
    if os.path.exists(object_path):
        try:
            obj = Mesh(object_path)
            if os.path.exists(_txt_path):
                obj.texture(_txt_path, scale=0.1)
            obj.lighting(style='off', ambient=0.2, diffuse=0.7, specular=0.8, specular_power=50)
            _plotter = show(obj, title=sub_dir, size=(1000, 1000), pos=(500, 0), interactive=False,
                            resetcam=True)
            cam_pos = _plotter.camera.GetPosition()
            # Take screenshot to detect 2d landmarks
            output_folder = os.path.join(os.path.dirname(object_path), '3D Landmarks')
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder, exist_ok=True)
            print(f"working on {sub_dir}...")
            # Frontal View
            _plotter.camera.SetPosition([0.0, 0.0, cam_pos[2]])
            _plotter.render()
            time.sleep(0.2)
            screenshot(os.path.join(output_folder, "s_shot.jpg"), scale=1)
            w, h = get_image_resolution(os.path.join(output_folder, "s_shot.jpg"))
            save_cam_state(_plotter.camera, os.path.join(output_folder, f"{sub_dir}_camera_front_view.txt"),
                           [w, h])
            # Left Profile View
            _plotter.camera.SetPosition([cam_pos[2], 0.0, 0.0])
            _plotter.render()
            time.sleep(0.2)
            screenshot(os.path.join(output_folder, "s_shot_left.jpg"), scale=1)
            w, h = get_image_resolution(os.path.join(output_folder, "s_shot_left.jpg"))
            save_cam_state(_plotter.camera, os.path.join(output_folder, f"{sub_dir}_camera_left_view.txt"),
                           [w, h])
            # Right Profile View
            _plotter.camera.SetPosition([-cam_pos[2], 0.0, 0.0])
            _plotter.render()
            time.sleep(0.2)
            screenshot(os.path.join(output_folder, "s_shot_right.jpg"), scale=1)
            w, h = get_image_resolution(os.path.join(output_folder, "s_shot_right.jpg"))
            save_cam_state(_plotter.camera, os.path.join(output_folder, f"{sub_dir}_camera_right_view.txt"),
                           [w, h])
            # Extract 2D landmarks from screenshots
            calculate_2d_landmarks(output_folder, output_folder)

            landmarks_3d = calculate_3d_landmarks(obj,
                                                  os.path.join(output_folder, "s_shot_landmarks_front.txt"),
                                                  _plotter,
                                                  os.path.join(output_folder,
                                                               f"{sub_dir}_camera_front_view.txt"))
            print(f"Front View 3D Landmarks Count: {len(landmarks_3d)}")
            with open(os.path.join(output_folder, f"{sub_dir}_3D_landmarks.txt"), "w") as file:
                for i, landmark in enumerate(landmarks_3d):
                    if i < 17:
                        continue
                    file.write(f"{landmark[0]}, {landmark[1]}, {landmark[2]}\n")

            landmarks_3d_left = calculate_3d_landmarks(obj, os.path.join(output_folder,
                                                                         "s_shot_left_landmarks_left_profile.txt"),
                                                       _plotter,
                                                       os.path.join(output_folder,
                                                                    f"{sub_dir}_camera_left_view.txt"))
            print(f"Left Profile View 3D Landmarks Count: {len(landmarks_3d_left)}")
            with open(os.path.join(output_folder, f"{sub_dir}_3D_landmarks.txt"), "a") as file:
                for i, landmark in enumerate(landmarks_3d_left):
                    if i > 8:
                        break
                    file.write(f"{landmark[0]}, {landmark[1]}, {landmark[2]}\n")

            landmarks_3d_right = calculate_3d_landmarks(obj, os.path.join(output_folder,
                                                                          "s_shot_right_landmarks_right_profile.txt"),
                                                        _plotter,
                                                        os.path.join(output_folder,
                                                                     f"{sub_dir}_camera_right_view.txt"))
            print(f"Right Profile View 3D Landmarks Count: {len(landmarks_3d_right)}")
            with open(os.path.join(output_folder, f"{sub_dir}_3D_landmarks.txt"), "a") as file:
                for i, landmark in enumerate(landmarks_3d_right):
                    if i > 8:
                        break
                    file.write(f"{landmark[0]}, {landmark[1]}, {landmark[2]}\n")
            _plotter.close()
            if remove_extra_files:
                remove_files_except_landmarks(output_folder)
        except Exception as ex:
            print(f"Error on extracting landmarks of : {sub_dir} : {ex}")
            try:
                _plotter.close()
            except Exception as exinex:
                print(f"Error on closing plotter : {sub_dir} : {exinex}")


def save_cam_state(cam, file_name, res):
    camera_dict = {
        "Position": cam.GetPosition(),
        "FocalPoint": cam.GetFocalPoint(),
        "ViewUp": cam.GetViewUp(),
        "Width": res[0],
        "Height": res[1]
    }
    camera_json = json.dumps(camera_dict, indent=4)
    output_file_path = file_name
    with open(output_file_path, "w") as json_file:
        json_file.write(camera_json)


def get_image_resolution(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    return width, height


def extract_from_custom_object(object_path):
    extract_landmarks(object_path, False)
    sub_dir = os.path.splitext(os.path.basename(object_path))[0]
    sub_dir_path = os.path.dirname(object_path)
    texture_path = os.path.join(sub_dir_path, f"{sub_dir}.jpg")
    if os.path.exists(object_path):
        obj = Mesh(object_path)
        if os.path.exists(texture_path):
            obj.texture(texture_path, scale=0.1)
        landmark_path = os.path.join(sub_dir_path, '3D Landmarks')
        visualize_3d_landmarks(
            obj,
            os.path.join(landmark_path, f"{sub_dir}_3D_landmarks.txt"),
            os.path.join(landmark_path, f"{sub_dir}_camera_front_view.txt"),
            sub_dir)
    else:
        print(object_path + " is not exist!")


def visualize_3d_landmarks(obj, landmarks_3d_file_path, camera_settings_file=None, window_title="", be_intractive=True):
    # Load 3D landmarks from the text file
    landmarks_3d = []
    with open(landmarks_3d_file_path, "r") as file:
        for line in file:
            x, y, z = map(float, line.strip().split(","))
            landmarks_3d.append([x, y, z])

    h = 1000
    w = 1000
    v_plotter = Plotter()
    if camera_settings_file:
        with open(camera_settings_file, "r") as json_file:
            camera_settings = json.load(json_file)
            v_plotter.camera.SetPosition(camera_settings["Position"])
            v_plotter.camera.SetFocalPoint(camera_settings["FocalPoint"])
            v_plotter.camera.SetViewUp(camera_settings["ViewUp"])
            w = int(camera_settings["Width"])
            h = int(camera_settings["Height"])

    obj.lighting(style='off', ambient=0.2, diffuse=0.7, specular=0.8, specular_power=50)
    v_plotter.show(obj, title=window_title, size=(w, h), interactive=False, resetcam=False)

    bounds = obj.bounds()
    obj_width = abs(bounds[1] - bounds[0])
    sphere_size = obj_width / 244.3466583887736

    cnt = 1
    for landmark in landmarks_3d:
        color = "red"
        sphere = Sphere(pos=landmark, r=sphere_size, c=color)
        v_plotter.at(0).add(sphere).render()
        cnt += 1
    if be_intractive:
        v_plotter.interactive()


def remove_files_except_landmarks(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not (file.endswith("_3D_landmarks.txt") or file.endswith("_camera_front_view.txt")):
                os.remove(os.path.join(root, file))


if __name__ == "__main__":

    command = input("The Dataset Path is : " + esrc_dataset_path + "\n"
                                                                   ""
                                                                   "Enter 1 to extract 3D landmarks for all ESRC "
                                                                   "dataset 3D objects\n"
                                                                   "Enter 2 to extract 3D landmarks for a 3D object ("
                                                                   "Custom Path)\n"
                                                                   "Enter 3 to visualize extracted landmarks of an "
                                                                   "ESRC 3D object\n\n").strip()
    if command == '1':
        extract_esrc_landmarks()
    if command == '2':
        obj_path = input("Please enter your custom object full path to visualize 3D landmarks\n")
        extract_from_custom_object(obj_path)
    elif command == '3':
        item = input("Please enter dataset item name to visualize 3D landmarks\n")
        path = os.path.join(esrc_dataset_path, item.strip(), f"{item.strip()}.obj")
        txt_path = os.path.join(esrc_dataset_path, item.strip(), f"{item.strip()}.jpg")
        if os.path.exists(path):
            obj = Mesh(path)
            if os.path.exists(txt_path):
                obj.texture(txt_path, scale=0.1)
            current_folder = os.path.join(esrc_dataset_path, item.strip(), '3D Landmarks')
            visualize_3d_landmarks(
                obj,
                os.path.join(current_folder, f"{item.strip()}_3D_landmarks.txt"),
                os.path.join(current_folder, f"{item.strip()}_camera_front_view.txt"),
                item.strip())
        else:
            print(path + " is not exist!")
