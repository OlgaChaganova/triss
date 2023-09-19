import cv2
import numpy as np
import onnxruntime as ort

from infra.exceptions import NoDetectionError
from infra.schema import BBox, DetectionResult


class PetDetectorYolov8:
    def __init__(
        self, onnx_model_path: str,
        confidence_thresh: float,
        iou_thresh: float,
    ) -> None:
        """Initializes an instance of the Yolov8 class.

        Args:
        ----
            onnx_model: Path to the ONNX model.
            confidence_thresh: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self._session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self._confidence_thresh = confidence_thresh
        self._iou_thresh = iou_thresh

        # Load the class names from the COCO dataset
        self._classes = {15: 'cat', 16: 'dog'}

        # Generate a color palette for the classes
        self._color_palette = {15: (0, 0, 255), 16: (0, 255, 0)}

        self._model_inputs = self._session.get_inputs()
        input_shape = self._model_inputs[0].shape
        self._input_width = input_shape[2]
        self._input_height = input_shape[3]

        self._img_height: int | None = None
        self._img_width: int | None = None

    def __call__(self, image: np.ndarray) -> DetectionResult:
        """Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns
        -------
            DetectionResult
            Boxes for all detected pets in the frame/image.
        """
        img_data = self._preprocess(image)
        outputs = self._session.run(None, {self._model_inputs[0].name: img_data})
        return self._postprocess(outputs)

    def draw_detections(self, image: np.ndarray, detection_results: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and labels on the input image based on the detected objects.

        Parameters
        ----------
        image: np.ndarray
            Input image / frame.
        detection_results: DetectionResult
            Result of object detection.

        Returns
        -------
            np.ndarray
            Input image with bbox, confidence and class label.
        """
        for detection in detection_results.result:
            x1, y1 = detection.top_left_x, detection.top_left_y
            x2, y2, = detection.bottom_right_x, detection.bottom_right_y

            # Retrieve the color for the class ID
            color = self._color_palette[detection.class_id]

            # Draw the bounding box on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Create the label text with class name and score
            label = f'{self._classes[detection.class_id]}: {detection.confidence:.2f}'

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                image,
                (label_x, label_y - label_height),
                (label_x + label_width, label_y + label_height),
                color,
                cv2.FILLED,
            )

            # Draw the label text on the image
            cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return image

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses the input image before performing inference.

        Parameters
        ----------
        image: np.ndarray
            Input raw image / frame.

        Returns
        -------
            np.ndarray
            Preprocessed image data ready for inference.
        """
        self._img_height, self._img_width = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._input_width, self._input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def _postprocess(self, output: np.ndarray) -> DetectionResult:
        """Perform post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Parameters
        ----------
        input_image: np.ndarray
            Input image or frame.
        output: np.ndarray
            Model output.

        Returns
        -------
            DetectionResult
            Bboxes for all detected pets in the frame or image.
        """
        det_result: list[BBox] = []
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self._img_width / self._input_width
        y_factor = self._img_height / self._input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self._confidence_thresh:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                if class_id not in self._classes:
                    continue

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self._confidence_thresh, self._iou_thresh)

        if len(indices) == 0:
            raise NoDetectionError

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            det_result.append(
                BBox(
                    top_left_x=box[0],
                    top_left_y=box[1],
                    bottom_right_x=(box[0] + box[2]),
                    bottom_right_y=(box[1] + box[3]),
                    confidence=score,
                    class_id=class_id,
                ),
            )

        return DetectionResult(result=det_result)
