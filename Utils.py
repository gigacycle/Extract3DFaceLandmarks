import cv2
import numpy as np
import torch
from PIL import Image
from PIL.Image import Transpose


def load_img_2_tensors(image_path, fa, face_detector, transform_func=None):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.copyMakeBorder(
        img,
        top=50,
        bottom=50,
        left=50,
        right=50,
        borderType=cv2.BORDER_DEFAULT
    )
    s = 1.5e3
    t = [0, 0, 0]
    scale = 1.2
    size = 256
    ds = face_detector.detect_from_image(img[..., ::-1].copy())
    x_s = 0
    y_s = 0
    x_e = 0
    y_e = 0
    center = [0, 0]
    for i in range(len(ds)):
        d = ds[i]
        center = [d[3] - (d[3] - d[1]) / 2.0, d[2] - (d[2] - d[0]) / 2.0]
        center[0] += (d[3] - d[1]) * 0.06
        center[0] = int(center[0])
        center[1] = int(center[1])
        l = max(d[2] - d[0], d[3] - d[1]) * scale
        if l < 200:
            continue
        x_s = center[1] - int(l / 2)
        y_s = center[0] - int(l / 2)
        x_e = center[1] + int(l / 2)
        y_e = center[0] + int(l / 2)
        t = [256. - center[1] + t[0], center[0] - 256. + t[1], 0]
        rescale = size / (x_e - x_s)
        s *= rescale
        t = [t[0] * rescale, t[1] * rescale, 0.]
        img = Image.fromarray(img).crop((x_s, y_s, x_e, y_e))
        img = cv2.resize(np.asarray(img), (size, size)).astype(np.float32)
        break
    assert img.shape[0] == img.shape[1] == 256
    ori_img_tensor = torch.from_numpy(img.transpose((2, 0, 1)).astype(np.float32) / 255.0)  # (C, H, W)
    img_tensor = ori_img_tensor.clone()
    if transform_func:
        img_tensor = transform_func(img_tensor)

    # Get 2D landmarks on image
    kpts_list = fa.get_landmarks(img)
    kpts = kpts_list[0]
    kpts_tensor = torch.from_numpy(kpts)  # (68, 2)

    return img_tensor, ori_img_tensor, kpts_tensor, [x_s, y_s, x_e, y_e], center


def trim_keypoints(keypoints, face_border_landmarks):
    # Define the face border polygon
    face_border_polygon = np.array(face_border_landmarks, dtype=np.int32)

    # List to store filtered keypoints
    filtered_keypoints = []

    for kp in keypoints:
        x, y = kp.pt

        # Check if the keypoint is inside the face border polygon
        if cv2.pointPolygonTest(face_border_polygon, (x, y), False) >= 0:
            filtered_keypoints.append(kp)

    return filtered_keypoints


def flip_horizontal_and_save(input_path, output_path):
    try:
        # Open the image
        img = Image.open(input_path)
        # Flip the image horizontally
        flipped_img = img.transpose(Transpose.FLIP_LEFT_RIGHT)
        # Save the flipped image to the specified path
        flipped_img.save(output_path)
        print("Image flipped and saved successfully.")
    except Exception as e:
        print("An error occurred:", e)


def flip_landmarks_tensor(landmarks, image_width):
    flipped_landmarks = np.copy(landmarks)
    flipped_landmarks[:, 0] = image_width - flipped_landmarks[:, 0]
    return torch.tensor(flipped_landmarks)


def flip_frame(frame, image_width):
    x0 = image_width - frame[2]
    x1 = image_width - frame[0]
    flipped_frame = [
        x0,  # Flipped x-coordinate of the left-top corner
        frame[1],  # Y-coordinate of the left-top corner (remains the same)
        x1,  # Flipped x-coordinate of the right-bottom corner
        frame[3]  # Y-coordinate of the right-bottom corner (remains the same)
    ]
    return flipped_frame


def flip_center(center, image_width):
    flipped_center = [image_width - center[0], center[1]]
    return flipped_center


def get_image_dimensions(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        return [width, height]
    except Exception as e:
        print("An error occurred:", e)
        return None
