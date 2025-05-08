import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import os


def normalised_ground_truth(matched_boxes, feature_box, return_format):
    """
    Normalizes ground truth boxes based on anchor box dimensions.
    For 'encode', returns the encoded representation.
    For 'decode', converts encoded predictions back to box coordinates.
    """
    matched_boxes = matched_boxes.astype(np.float32)
    feature_box = feature_box.astype(np.float32)

    if return_format == "encode":
        x_offset = (matched_boxes[:, 0] - feature_box[:, 0]) / feature_box[:, 2]
        y_offset = (matched_boxes[:, 1] - feature_box[:, 1]) / feature_box[:, 3]
        w_scale = np.log(matched_boxes[:, 2] / feature_box[:, 2])
        h_scale = np.log(matched_boxes[:, 3] / feature_box[:, 3])
        encoded_boxes = np.stack([x_offset, y_offset, w_scale, h_scale], axis=-1)
        scale_factors = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
        return encoded_boxes / scale_factors

    elif return_format == "decode":
        scale_factors = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
        encoded_boxes = matched_boxes * scale_factors
        x_center = encoded_boxes[:, 0] * feature_box[:, 2] + feature_box[:, 0]
        y_center = encoded_boxes[:, 1] * feature_box[:, 3] + feature_box[:, 1]
        w = np.exp(encoded_boxes[:, 2]) * feature_box[:, 2]
        h = np.exp(encoded_boxes[:, 3]) * feature_box[:, 3]
        decoded_boxes = np.stack([x_center, y_center, w, h], axis=-1)
        return decoded_boxes

    else:
        raise ValueError("return_format must be either 'encode' or 'decode'")
    

def detect_crop(image_path, priors,detection_session, detection_input_name, detection_output_names):
    """
    Combines object detection and label inference:
      1. Reads and preprocesses the image.
      2. Runs SSD detection using ONNX.
      3. Decodes the bounding boxes.
      4. Draws bounding boxes on the image.
      5. Displays the annotated image and returns the results.
    """
    # Read the original image (for cropping & display) and a normalized version for detection.
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Unable to load image at {image_path}")
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(orig_img, (300, 300))  # This matches detection model input

    # Normalize image for detection inference
    img_norm = resized_img.astype(np.float32) / 255.0
    image_input = np.expand_dims(img_norm, axis=0)

    # Run detection model
    predictions = detection_session.run(detection_output_names, {detection_input_name: image_input})
    model_predictions = predictions[0]

    # Decode bounding boxes using the loaded priors
    boxes = normalised_ground_truth(model_predictions[0][:, :4], np.array(priors), 'decode')
    # Get confidence scores (assumed to be at index 4)
    classes = model_predictions[..., 4][0]
    
    # Select a subset of boxes based on score (your original logic)
    idx = np.argsort(classes)[:1]
    boxes = boxes[idx] * 1024

    # Convert from center-based boxes [x_center, y_center, w, h] to [xmin, ymin, xmax, ymax]
    xmin = boxes[:, 0] - boxes[:, 2] / 2
    ymin = boxes[:, 1] - boxes[:, 3] / 2
    xmax = boxes[:, 0] + boxes[:, 2] / 2
    ymax = boxes[:, 1] + boxes[:, 3] / 2
    fp = np.vstack((xmin, ymin, xmax, ymax)).T

    image_display = cv2.resize(orig_img, (1024,1024))
    x_min, y_min, x_max, y_max = map(int, fp[0])

    crop = image_display[y_min:y_max, x_min:x_max]

    return crop

def decode_batch_predictions_numpy(
    probs: np.ndarray,
    idx_to_char: list[str],
    blank_id: int = None,
    max_len: int = 32
) -> list[str]:
    """
    probs: shape [B, T, C_out] softmax scores
    idx_to_char: list of length C_out−1 mapping class IDs → chars
    blank_id: ID of the CTC blank (defaults to C_out−1)
    """
    B, T, C = probs.shape
    if blank_id is None:
        blank_id = C - 1

    # 1) Greedy best‐path
    best_path = np.argmax(probs, axis=2)  # shape [B, T]

    decoded_texts = []
    for seq in best_path:
        out_chars = []
        prev_id   = None
        for idx in seq:
            # collapse repeats
            if idx == prev_id:
                continue
            prev_id = idx
            # drop blanks
            if idx == blank_id:
                continue
            # map to char
            out_chars.append(idx_to_char[idx])
        decoded_texts.append("".join(out_chars)[:max_len])

    return decoded_texts


def visualize_predictions_onnx_from_folder(image,ort_session,idx_to_char, blank_id,target_size=(256,128),plot=True):

    img = cv2.resize(image, target_size)/255
    img = np.expand_dims(img,axis=0)

    input_name = ort_session.get_inputs()[0].name
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    preds  = ort_session.run(None, {input_name: img})[0]                            # [B, T, C_out]
    texts  = decode_batch_predictions_numpy(preds, idx_to_char, blank_id)
    
    print(texts)

    if plot:
        plt.imshow(img[0])
        plt.tight_layout()
        plt.show()