import os
import sys
import subprocess
import urllib.request
import torch
import cv2
import numpy as np
import supervision as sv
import re
from typing import List, Tuple, Optional
from PIL import Image


box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}\n{stderr.decode('utf-8')}")
    else:
        print(stdout.decode('utf-8'))

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    try:
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

class GroundingDINOPostProcessor:
    def _init_(self, confidence_threshold=0.25, nms_threshold=0.6, min_box_area=100, max_box_area_ratio=0.8, context_expansion_ratio=0.2):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_box_area = min_box_area
        self.max_box_area_ratio = max_box_area_ratio
        self.context_expansion_ratio = context_expansion_ratio
    
    def calculate_iou(self, box1, box2):
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        intersection_area = (x2_min - x1_max) * (y2_min - y1_max)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area if union_area > 0 else 0.0

    def parse_query_context(self, query):
        query_lower = query.lower()
        relationships = {
            'person_with_object': any(p in query_lower for p in [
                'person with', 'man with', 'woman with', 'person holding', 
                'man holding', 'woman holding', 'person using', 'person carrying'
            ]),
            'person_doing_action': any(p in query_lower for p in [
                'person doing', 'person playing', 'person working', 'person sitting',
                'person standing', 'person walking', 'person running'
            ]),
            'interaction': any(p in query_lower for p in [
                'talking to', 'selling to', 'buying from', 'giving to', 'showing to'
            ])
        }
        entities = {
            'people_terms': ['person', 'man', 'woman', 'child', 'people', 'individual', 'human'],
            'object_terms': self.extract_keywords_from_query(query)
        }
        return {'relationships': relationships, 'entities': entities}
    
    def find_contextual_boxes(self, all_boxes, all_scores, all_labels, query):
        context_info = self.parse_query_context(query)
        if context_info['relationships']['person_with_object']:
            person_boxes, object_boxes = [], []
            person_scores, object_scores = [], []
            for box, score, label in zip(all_boxes, all_scores, all_labels):
                label_lower = label.lower()
                if any(t in label_lower for t in context_info['entities']['people_terms']):
                    person_boxes.append(box)
                    person_scores.append(score)
                elif any(t in label_lower for t in context_info['entities']['object_terms']):
                    object_boxes.append(box)
                    object_scores.append(score)
            if person_boxes and object_boxes:
                return self.combine_related_boxes(person_boxes, object_boxes, person_scores, object_scores)
        if len(all_boxes) > 0:
            best_idx = np.argmax(all_scores)
            return all_boxes[best_idx], all_scores[best_idx]
        return None, 0.0
    
    def combine_related_boxes(self, person_boxes, object_boxes, person_scores, object_scores):
        best_score = 0
        best_combined_box = None
        for person_box, person_score in zip(person_boxes, person_scores):
            for object_box, object_score in zip(object_boxes, object_scores):
                spatial_score = self.calculate_spatial_relationship(person_box, object_box)
                if spatial_score > 0.1:
                    combined_box = self.create_combined_box(person_box, object_box)
                    combined_score = (person_score + object_score) / 2 * spatial_score
                    if combined_score > best_score:
                        best_score = combined_score
                        best_combined_box = combined_box
        if best_combined_box is None and person_boxes:
            best_idx = np.argmax(person_scores)
            return person_boxes[best_idx], person_scores[best_idx]
        return best_combined_box, best_score
    
    def calculate_spatial_relationship(self, box1, box2):
        center1 = np.array([(box1[0] + box1[2])/2, (box1[1] + box1[3])/2])
        center2 = np.array([(box2[0] + box2[2])/2, (box2[1] + box2[3])/2])
        distance = np.linalg.norm(center1 - center2)
        avg_size = (np.sqrt((box1[2] - box1[0]) * (box1[3] - box1[1])) + np.sqrt((box2[2] - box2[0]) * (box2[3] - box2[1]))) / 2
        if avg_size > 0:
            normalized_distance = distance / avg_size
            spatial_score = max(0, 1 - normalized_distance / 3)
        else:
            spatial_score = 0
        iou = self.calculate_iou(box1, box2)
        if iou > 0.1:
            spatial_score += 0.3
        return min(spatial_score, 1.0)
    
    def create_combined_box(self, box1, box2):
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return np.array([x1, y1, x2, y2])
    
    def expand_box_for_context(self, box, image_shape):
        h, w = image_shape
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        expand_x = box_w * self.context_expansion_ratio
        expand_y = box_h * self.context_expansion_ratio
        expanded_box = np.array([
            max(0, box[0] - expand_x),
            max(0, box[1] - expand_y),
            min(w, box[2] + expand_x),
            min(h, box[3] + expand_y)
        ])
        return expanded_box

    def extract_keywords_from_query(self, query):
        stop_words = {'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        clean_query = re.sub(r'[^\w\s]', '', query.lower())
        return [word for word in clean_query.split() if word not in stop_words and len(word) > 2]

    def calculate_relevance_score(self, box, confidence, query, label, image_shape):
        relevance_score = confidence * 0.7
        query_lower = query.lower()
        label_lower = label.lower()
        if label_lower in query_lower or any(word in query_lower for word in label_lower.split()):
            relevance_score += 0.2
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        image_area = image_shape[0] * image_shape[1]
        area_ratio = box_area / image_area
        if 0.05 <= area_ratio <= 0.5:
            area_score = min(area_ratio * 2, 0.4 - abs(area_ratio - 0.2))
            relevance_score += area_score * 0.1
        return relevance_score

class CLIPReranker:
    def _init_(self, device="cpu"):
        self.device = device
        self.model = None
        self.preprocess = None
        self._load_clip()
    
    def _load_clip(self):
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as e:
            print(f"CLIP not loaded: {e}")
    
    def is_available(self):
        return self.model is not None
    
    def crop_and_preprocess(self, image, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cropped = image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)
        return self.preprocess(pil_image).unsqueeze(0).to(self.device)
    
    def calculate_clip_similarity(self, image, boxes, query):
        if not self.is_available() or len(boxes) == 0:
            return np.array([])
        import clip
        text_tokens = clip.tokenize([query]).to(self.device)
        similarities = []
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            for box in boxes:
                try:
                    image_tensor = self.crop_and_preprocess(image, box)
                    image_features = self.model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    similarities.append((text_features @ image_features.T).item())
                except:
                    similarities.append(0.0)
        return np.array(similarities)

# class EnhancedGroundingDINOPostProcessor:
#     # def _init_(self, use_clip=True, clip_weight=0.4):
#     #     self.base_processor = GroundingDINOPostProcessor()
#     #     self.use_clip = use_clip
#     #     self.clip_weight = clip_weight
#     #     self.clip_reranker = CLIPReranker()
#     def __init__(self, use_clip_reranking=False):
#         self.use_clip_reranking = use_clip_reranking if use_clip_reranking else None
    
#     def process_detections(self, detections, query, image):
#         if len(detections.xyxy) == 0:
#             return None
#         image_shape = image.shape[:2]
#         boxes = detections.xyxy
#         scores = detections.confidence
#         filtered_indices = []
#         for i, (box, score) in enumerate(zip(boxes, scores)):
#             if score < self.base_processor.confidence_threshold:
#                 continue
#             box_area = (box[2] - box[0]) * (box[3] - box[1])
#             image_area = image_shape[0] * image_shape[1]
#             if (box_area < self.base_processor.min_box_area or box_area > (self.base_processor.max_box_area_ratio * image_area)):
#                 continue
#             filtered_indices.append(i)
#         if not filtered_indices:
#             return None
#         filtered_boxes = boxes[filtered_indices]
#         filtered_scores = scores[filtered_indices]
#         filtered_detections = sv.Detections(
#             xyxy=filtered_boxes,
#             confidence=filtered_scores,
#             class_id=np.zeros(len(filtered_boxes), dtype=int)
#         )
#         nms_detections = filtered_detections.with_nms(threshold=self.base_processor.nms_threshold)
#         if len(nms_detections.xyxy) == 0:
#             return None
#         if self.use_clip and self.clip_reranker and self.clip_reranker.is_available():
#             clip_similarities = self.clip_reranker.calculate_clip_similarity(image, nms_detections.xyxy, query)
#             if len(clip_similarities) > 0:
#                 combined_scores = ((1 - self.clip_weight) * nms_detections.confidence + self.clip_weight * clip_similarities)
#                 best_idx = np.argmax(combined_scores)
#                 final_score = combined_scores[best_idx]
#             else:
#                 best_idx = np.argmax(nms_detections.confidence)
#                 final_score = nms_detections.confidence[best_idx]
#         else:
#             best_idx = np.argmax(nms_detections.confidence)
#             final_score = nms_detections.confidence[best_idx]
#         best_box = nms_detections.xyxy[best_idx]
#         return best_box, final_score
class EnhancedGroundingDINOPostProcessor:
    def __init__(self, use_clip_reranking=False):
        self.use_clip_reranking = use_clip_reranking
        # --- FIX: Define the necessary thresholds directly in this class ---
        self.confidence_threshold = 0.35
        self.min_box_area = 100
        self.max_box_area_ratio = 0.8

    def process_detections(self, detections, query, image):
        if len(detections.xyxy) == 0:
            return None

        image_shape = image.shape[:2]
        boxes = detections.xyxy
        scores = detections.confidence
        filtered_indices = []

        for i, (box, score) in enumerate(zip(boxes, scores)):
            # --- FIX: Use the threshold from this class ---
            if score < self.confidence_threshold:
                continue

            box_area = (box[2] - box[0]) * (box[3] - box[1])
            image_area = image_shape[0] * image_shape[1]
            
            # --- FIX: Use the area properties from this class ---
            if (box_area < self.min_box_area or box_area > (self.max_box_area_ratio * image_area)):
                continue
            
            filtered_indices.append(i)

        if not filtered_indices:
            return None

        detections.xyxy = detections.xyxy[filtered_indices]
        detections.confidence = detections.confidence[filtered_indices]
        
        # Add any other attributes that need filtering
        if detections.class_id is not None:
             detections.class_id = detections.class_id[filtered_indices]

        return detections
def save_cropped_region(image, box, query, output_dir):
    x1, y1, x2, y2 = map(int, box)
    cropped_region = image[y1:y2, x1:x2]
    safe_query = re.sub(r'[^\w\s-]', '', query).strip()
    safe_query = re.sub(r'[-\s]+', '_', safe_query)
    cropped_path = os.path.join(output_dir, f"cropped_{safe_query}.jpg")
    cv2.imwrite(cropped_path, cropped_region)
    
    return cropped_path

try:
    HOME = os.getcwd()
    from groundingdino.util.inference import Model as GroundingDINOModel
    DEVICE = torch.device('cpu')
    weights_dir = os.path.join(HOME, "weights")
    grounding_dino_weights_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
    if not os.path.exists(grounding_dino_weights_path):
        download_file("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", grounding_dino_weights_path)
    GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    grounding_dino_model = GroundingDINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=grounding_dino_weights_path, device=str(DEVICE))

    image_input = input("Enter image path (default: data/img.png): ").strip()
    if not image_input:
        data_dir = os.path.join(HOME, "data")
        os.makedirs(data_dir, exist_ok=True)
        SOURCE_IMAGE_PATH = os.path.join(data_dir, "img.png")
    else:
        SOURCE_IMAGE_PATH = os.path.join(HOME, image_input) if not os.path.isabs(image_input) else image_input
    if not os.path.exists(SOURCE_IMAGE_PATH):
        sys.exit("Image not found.")

    QUERY = input("Enter your query: ").strip()
    USE_CLIP_RERANKING = input("Use CLIP re-ranking? (y/n, default=y): ").strip().lower() in ['', 'y', 'yes']

    image = cv2.imread(SOURCE_IMAGE_PATH)
    processor = EnhancedGroundingDINOPostProcessor(use_clip_reranking=USE_CLIP_RERANKING)
    
    phrases = [phrase.strip() for phrase in QUERY.split('.')]
    expanded_classes = phrases + ['person', 'people', 'man', 'woman', 'child', 'human', 'individual']
    all_classes = [QUERY] + list(set(expanded_classes) - {QUERY})

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=all_classes,
        box_threshold=0.25,
        text_threshold=0.20
    )
    result = processor.process_detections(detections, QUERY, image)
    # --- FIX: Create the results directory right after processing ---
    os.makedirs("results", exist_ok=True)

# Initialize variables to a default state
    best_box = None
    final_score = 0.0

# Process the results if any were found
    if result is not None and len(result.xyxy) > 0:
        # Find the index of the box with the highest confidence score
        best_index = result.confidence.argmax()
        # Get the coordinates and score of the best box
        best_box = result.xyxy[best_index]
        final_score = result.confidence[best_index]

    # --- Annotation ---
    # Create annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

# Check if a best box was found before annotating
    # Create a Detections object for the single best result
    if best_box is not None:
        best_detection = sv.Detections(
        xyxy=np.array([best_box]),
        confidence=np.array([final_score]),
        class_id=np.array([0])
        )

    # Create the label for the best detection
        labels = [f"{QUERY} ({final_score:.2f})"]

    # Annotate the scene with the box and the label
        annotated_image = box_annotator.annotate(
            scene=image.copy(),
            detections=best_detection
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=best_detection,
            labels=labels
        )

    # Save the annotated image
        cv2.imwrite("results/annotated_image.jpg", annotated_image)

    # Crop the image to the best bounding box
        x1, y1, x2, y2 = map(int, best_box)
        cropped_image = image[y1:y2, x1:x2]
        cv2.imwrite("results/cropped_image.jpg", cropped_image)

        print(f"Processing complete. Best box found with confidence {final_score:.2f}.")
        print("Results saved in the 'results' folder.")

    else:
    # If no box was found, just save the original image
        annotated_image = image.copy()
        cv2.imwrite("results/annotated_image.jpg", annotated_image)
        print("No objects matching the query were found in the image.")

except Exception as e:
    import traceback
    traceback.print_exc()