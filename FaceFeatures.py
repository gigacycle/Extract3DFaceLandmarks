import os
from pathlib import Path
# External libs
import cv2
import numpy as np
import torch
import torchvision

import Utils
# Internal libs
from Utils import flip_center,flip_frame,flip_landmarks_tensor,flip_horizontal_and_save,load_img_2_tensors,get_image_dimensions,trim_keypoints


def preprocess(img_dir, output_dir, fa, face_detector, save, device='cuda'):
    """
    Propare data for inferencing.
    img_dir: directory of input images. str.
    fa: face alignment model. From https://github.com/1adrianb/face-alignment
    face_detector: face detector model. From https://github.com/1adrianb/face-alignment
    """
    transform_func = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    img_list = Path(img_dir).glob('*.jpg')

    img_tensors = []
    ori_img_tensors = []
    kpts_tensors = []
    if not output_dir.endswith('/'):
        output_dir += '/'
    for image_name in img_list:
        try:
            image_path = image_name.__str__()
            ext = os.path.splitext(image_name)
            f_name = os.path.basename(image_name)
            f_name_no_ext = f_name.replace(ext[1], '')
            savePath = output_dir + f_name_no_ext + '_landmarks.txt'
            outputImagePath = output_dir + f_name_no_ext + '_landmarked.jpg'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print("Calculating landmarks for : " + f_name)
            flip = False
            try:
                img_tensor, ori_img_tensor, kpts_tensor, frame, center = load_img_2_tensors(image_path, fa,face_detector,transform_func)
            except Exception:
                Utils.flip_horizontal_and_save(image_path, output_dir + f_name_no_ext + '_flipped.jpg')
                img_tensor, ori_img_tensor, kpts_tensor, frame, center = load_img_2_tensors(output_dir + f_name_no_ext + '_flipped.jpg', fa, face_detector, transform_func)
                os.remove(output_dir + f_name_no_ext + '_flipped.jpg')
                dim = Utils.get_image_dimensions(image_path)
                # kpts_tensor = Utils.flip_landmarks_tensor(kpts_tensor, 256)
                frame = Utils.flip_frame(frame, dim[0])
                center = Utils.flip_center(center, dim[0])
                flip = True

            kpts_array = kpts_tensor.numpy()
            img_tensors.append(img_tensor)
            ori_img_tensors.append(ori_img_tensor)
            kpts_tensors.append(kpts_tensor)
            if save:
                # Convert ori_img_tensor and img_tensor to NumPy arrays
                ori_img_array = (ori_img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                mark_landmarks_on_image(savePath, image_path, outputImagePath, kpts_array, frame, ori_img_array, flip)
        except Exception as error:
            print("An exception occurred while processing " + os.path.basename(
                image_name) + " because " + error.__str__())

    img_tensors = torch.stack(img_tensors, dim=0).unsqueeze(0)  # (1, V, C, H, W)
    ori_img_tensors = torch.stack(ori_img_tensors, dim=0).unsqueeze(0)  # (1, V, C, H, W)
    kpts_tensors = torch.stack(kpts_tensors, dim=0).unsqueeze(0)  # (1, V, 68, 2)

    if device == 'cuda':
        return img_tensors.cuda(), ori_img_tensors.cuda(), kpts_tensors.cuda()
    elif device == 'cpu':
        return img_tensors.cpu(), ori_img_tensors.cpu(), kpts_tensors.cpu()
    else:
        return ''


def mark_landmarks_on_image(savePath, image_path, outputPth, landmarks, frame, ori_img_array, isFlipped):
    # Load the image
    image1 = cv2.imread(image_path)

    added_margin_border_x = 50
    if isFlipped:
        added_margin_border_x = 0
    added_margin_border_y = 50

    h, w = image1.shape[:2]

    # Adjust the cropping coordinates to include additional pixels
    x1 = max(frame[0], 0)  # Left coordinate
    y1 = max(frame[1], 0)  # Top coordinate
    x2 = min(frame[2], image1.shape[1])  # Right coordinate
    y2 = min(frame[3], image1.shape[0])  # Bottom coordinate

    # Calculate the additional pixels needed on the image sides
    additional_pixels_left = max(-frame[0], 0)
    additional_pixels_top = max(-frame[1], 0)
    additional_pixels_right = max(frame[2] - w, 0)
    additional_pixels_bottom = max(frame[3] - h, 0)

    x1 -= additional_pixels_left
    x2 += additional_pixels_right
    y1 -= additional_pixels_top
    y2 += additional_pixels_bottom

    # Calculate the width and height of the cropped region
    width1 = x2 - x1
    height1 = y2 - y1

    newLandmarks = []
    # Convert the landmarks to integer coordinates
    landmarks = landmarks.astype(np.int32)
    height2, width2 = ori_img_array.shape[:2]
    # Draw circles at the landmark positions
    cnt = 10
    selected_landmarks, face_view = select_landmarks_for_reconstruction(landmarks)
    if isFlipped:
        if 'left' in face_view:
            face_view = face_view.replace('left', 'right')
        elif 'right' in face_view:
            face_view = face_view.replace('right', 'left')
    image1 = cv2.putText(image1, face_view, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    savePath = savePath.replace('.txt', '_{}.txt'.format(face_view))
    for landmark in selected_landmarks:
        x, y = landmark
        if isFlipped:
            x = width2 - x + (1 * added_margin_border_y / 3)
        # cv2.circle(ori_img_array, (x, y), 2, (0,255,0), -1)
        newX = round((x * width1) / width2) + x1 - added_margin_border_x
        newY = round((y * height1) / height2) + y1 - added_margin_border_y
        cv2.circle(image1, (newX, newY), 2, (0, 255 - cnt, cnt), -1)
        image1 = cv2.putText(image1, str(int(cnt / 10)), (newX, newY + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1,
                             cv2.LINE_AA)
        newLandmarks.append([newX, newY])
        cnt += 10

    # Save the marked image to a new file
    # cv2.imwrite(outputPth2, ori_img_array)
    cv2.imwrite(outputPth, image1)
    newLandmarks = np.array(newLandmarks)
    np.savetxt(savePath, newLandmarks, delimiter=',', fmt='%.0f')


def select_landmarks_for_reconstruction(landmarks):
    l_face_border = landmarks[0:9]
    l_eyebrow = landmarks[17:22]
    l_eye = landmarks[36:42]
    l_subnasal = landmarks[31:34]
    l_lips = np.concatenate((landmarks[48:52], landmarks[57:63], landmarks[66:68]))
    nose = landmarks[27:31]
    r_face_border = landmarks[8:17]
    r_eyebrow = landmarks[22:27]
    r_eye = landmarks[42:48]
    r_subnasal = landmarks[33:36]
    r_lips = np.concatenate((landmarks[51:58], landmarks[62:67]))

    # Identify the face view based on landmark positions
    left_eye_corner = np.min(l_eye, axis=0)
    right_eye_corner = np.max(r_eye, axis=0)
    nose_bridge = np.mean(nose, axis=0)
    nose_tip = landmarks[30]

    # Compute distances between key landmarks
    left_eye_to_nose = np.linalg.norm(left_eye_corner - nose_bridge)
    right_eye_to_nose = np.linalg.norm(right_eye_corner - nose_bridge)
    nose_tip_to_nose = np.linalg.norm(nose_tip - nose_bridge)

    # Determine the face view based on distances

    if left_eye_corner[0] > nose_bridge[0]:
        face_view = "left_profile"
    elif right_eye_corner[0] < nose_bridge[0]:
        face_view = "right_profile"
    else:
        face_view = "front"

    # Select relevant landmarks based on the inferred face view
    if face_view == "front":
        selected_landmarks = landmarks
    elif face_view == "left_profile":
        selected_landmarks = np.concatenate((r_face_border, r_eyebrow, r_eye, nose, r_subnasal, r_lips))
    elif face_view == "right_profile":
        selected_landmarks = np.concatenate((l_face_border, l_eyebrow, l_eye, nose, l_subnasal, l_lips))

    return selected_landmarks, face_view

