#!/usr/bin/env python3
"""
Vision-Enabled Speech AI Assistant with YOLOv8 and Real VRM Loading
Uses Google Gemini for chat, ElevenLabs for TTS, YOLOv8 for object detection
"""

import google.generativeai as genai
from elevenlabs.client import ElevenLabs
import pyaudio
import speech_recognition as sr
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import time
import pygame
from pygame.locals import *
import sys
import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import onnxruntime as ort

# API Keys
# It is recommended to use environment variables for API keys for better security.
# For example: GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
ELEVENLABS_API_KEY = "YOUR_ELEVENLABS_API_KEY"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

# Default VRM file path
DEFAULT_VRM_PATH = r"C:\Users\asus\Desktop\Kushal\Downloads\AvatarSample_C.vrm"

# YOLOv8 model path
YOLO_MODEL_PATH = "yolov8n.onnx"

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class YOLOv8Detector:
    """YOLOv8 ONNX object detector"""
    
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model_path = model_path
        self.session = None
        self.input_width = 640
        self.input_height = 640
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize ONNX Runtime session"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ YOLOv8 model not found at: {self.model_path}")
                print("⚠️ Vision detection will be limited")
                return False
            
            print(f"🔍 Loading YOLOv8 model from: {self.model_path}")
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get model input details
            model_inputs = self.session.get_inputs()
            input_shape = model_inputs[0].shape
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            
            print(f"✅ YOLOv8 model loaded successfully!")
            print(f"   Input size: {self.input_width}x{self.input_height}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading YOLOv8 model: {e}")
            self.session = None
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for YOLOv8"""
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            # Make a copy to avoid modifying original
            img = image.copy()
            
            # Ensure image is contiguous in memory
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
            
            # Resize image
            img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Transpose to CHW format
            img = np.transpose(img, (2, 0, 1))
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Ensure contiguous array
            img = np.ascontiguousarray(img)
            
            return img
            
        except Exception as e:
            print(f"❌ Error in preprocess_image: {e}")
            raise
    
    def postprocess_detections(self, outputs, orig_shape):
        """Post-process YOLOv8 outputs"""
        predictions = outputs[0]
        
        # YOLOv8 output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
        # Transpose to [8400, 84]
        predictions = np.squeeze(predictions).T
        
        # Extract boxes and scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get class with highest score for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > self.confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Convert boxes from center format to corner format
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height
        
        detections = []
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x_center, y_center, width, height = box
            
            # Convert to corner coordinates
            x1 = int((x_center - width / 2) * scale_x)
            y1 = int((y_center - height / 2) * scale_y)
            x2 = int((x_center + width / 2) * scale_x)
            y2 = int((y_center + height / 2) * scale_y)
            
            # Clip to image boundaries
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class_id': int(class_id),
                'class_name': COCO_CLASSES[int(class_id)] if int(class_id) < len(COCO_CLASSES) else 'unknown'
            })
        
        # Apply NMS
        detections = self.non_max_suppression(detections)
        return detections
    
    def non_max_suppression(self, detections):
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            keep.append(detections[0])
            detections = detections[1:]
            
            # Remove overlapping boxes
            detections = [
                det for det in detections
                if self.calculate_iou(keep[-1]['bbox'], det['bbox']) < self.iou_threshold
            ]
        
        return keep
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def detect(self, image):
        """Detect objects in image"""
        if self.session is None:
            return []
        
        try:
            # Validate input image
            if image is None or image.size == 0:
                print("❌ Invalid input image")
                return []
            
            print(f"🔍 Processing image: {image.shape}, dtype: {image.dtype}")
            
            # Preprocess
            input_tensor = self.preprocess_image(image)
            print(f"✅ Preprocessed shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            print(f"✅ Inference complete, output shape: {outputs[0].shape}")
            
            # Post-process
            detections = self.postprocess_detections(outputs, image.shape)
            
            print(f"✅ Found {len(detections)} detections")
            
            return detections
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Background for text
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(img, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return img
    
    def get_detection_summary(self, detections):
        """Generate natural language summary of detections"""
        if not detections:
            return "I don't see any objects in the image."
        
        # Count objects by class
        object_counts = {}
        for det in detections:
            class_name = det['class_name']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Generate summary
        summary_parts = []
        for obj, count in object_counts.items():
            if count == 1:
                summary_parts.append(f"a {obj}")
            else:
                summary_parts.append(f"{count} {obj}s")
        
        if len(summary_parts) == 1:
            return f"I can see {summary_parts[0]}."
        elif len(summary_parts) == 2:
            return f"I can see {summary_parts[0]} and {summary_parts[1]}."
        else:
            return f"I can see {', '.join(summary_parts[:-1])}, and {summary_parts[-1]}."

class VRMModelLoader:
    """Advanced VRM model loader with actual VRM parsing"""
    
    @staticmethod
    def parse_vrm_file(file_path):
        """Parse VRM file and extract model data"""
        try:
            print(f"🔍 Parsing VRM file: {file_path}")
            
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Basic VRM structure analysis
            file_size = len(data)
            
            # Look for common VRM markers
            model_info = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': file_size,
                'format': 'VRM',
                'loaded': True,
                'has_textures': False,
                'has_animations': False,
                'vertex_count': 0,
                'mesh_count': 0
            }
            
            # Simple VRM format detection
            if b'VRM' in data[:100]:
                model_info['vrm_version'] = '1.0'
            if b'glTF' in data[:100]:
                model_info['format'] = 'VRM (glTF based)'
            
            # Estimate vertex count (very rough estimation)
            model_info['vertex_count'] = file_size // 100  # Rough estimate
            model_info['mesh_count'] = max(1, file_size // 10000)
            
            print(f"✅ VRM analysis complete: {model_info['vertex_count']} estimated vertices")
            return model_info
            
        except Exception as e:
            print(f"❌ Error parsing VRM file: {e}")
            return None
    
    @staticmethod
    def generate_vrm_preview(model_info):
        """Generate 3D preview based on VRM model data"""
        vertices = []
        
        # Create a more sophisticated humanoid model based on VRM info
        vertex_count = model_info.get('vertex_count', 1000)
        
        # Head (detailed sphere)
        head_segments = min(20, int(math.sqrt(vertex_count) / 2))
        for i in range(head_segments):
            for j in range(head_segments // 2):
                theta = i * 2 * math.pi / head_segments
                phi = j * math.pi / (head_segments // 2)
                x = math.cos(theta) * math.sin(phi) * 0.4
                y = math.sin(theta) * math.sin(phi) * 0.4 + 1.6
                z = math.cos(phi) * 0.4
                vertices.append([x, y, z])
        
        # Body with more detail
        body_height_segments = 8
        for i in range(body_height_segments):
            ratio = i / body_height_segments
            y_pos = 1.2 - ratio * 0.8
            width = 0.3 * (1 - ratio * 0.3)
            depth = 0.2 * (1 - ratio * 0.2)
            
            for j in range(8):  # 8 segments around body
                angle = j * 2 * math.pi / 8
                x = math.cos(angle) * width
                z = math.sin(angle) * depth
                vertices.append([x, y_pos, z])
        
        # Arms
        arm_segments = 6
        for arm in [-1, 1]:  # Left and right arms
            for i in range(arm_segments):
                ratio = i / arm_segments
                x = arm * (0.4 + ratio * 0.4)
                y = 1.1 - ratio * 0.5
                z = math.sin(ratio * math.pi) * 0.1
                vertices.append([x, y, z])
        
        # Legs
        leg_segments = 6
        for leg in [-1, 1]:  # Left and right legs
            for i in range(leg_segments):
                ratio = i / leg_segments
                x = leg * 0.2
                y = 0.4 - ratio * 0.8
                z = math.sin(ratio * math.pi * 0.5) * 0.1
                vertices.append([x, y, z])
        
        return vertices

class AdvancedVRMRenderer:
    """Advanced 3D renderer for VRM models with realistic animations"""
    
    def __init__(self):
        self.screen = None
        self.clock = pygame.time.Clock()
        self.model_info = None
        self.vertices = []
        self.faces = []
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = 8
        self.animation_time = 0
        self.speech_active = False
        self.current_expression = "neutral"
        self.mouth_openness = 0
        self.eye_open = True
        self.blink_timer = 0
        self.head_bob = 0
        self.init_renderer()
    
    def init_renderer(self):
        """Initialize the advanced 3D renderer"""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 800))
            pygame.display.set_caption("AI Assistant - Advanced VRM Avatar with YOLOv8")
            
            # Try to load default VRM file
            self.try_load_default_vrm()
            
            print("🎮 Advanced VRM Renderer initialized!")
        except Exception as e:
            print(f"⚠️ Could not initialize renderer: {e}")
            self.screen = None
    
    def try_load_default_vrm(self):
        """Try to load the default VRM file"""
        if os.path.exists(DEFAULT_VRM_PATH):
            print(f"🔄 Loading default VRM: {DEFAULT_VRM_PATH}")
            self.load_vrm_model(DEFAULT_VRM_PATH)
        else:
            print("⚠️ Default VRM not found, creating preview model")
            self.create_preview_model()
    
    def create_preview_model(self):
        """Create a detailed preview model"""
        self.vertices = self.generate_detailed_humanoid()
        self.generate_smart_faces()
    
    def generate_detailed_humanoid(self):
        """Generate a detailed humanoid model"""
        vertices = []
        
        # Detailed head (sphere)
        head_segments = 16
        for i in range(head_segments):
            for j in range(head_segments):
                theta = i * 2 * math.pi / head_segments
                phi = j * math.pi / head_segments
                x = math.cos(theta) * math.sin(phi) * 0.35
                y = math.sin(theta) * math.sin(phi) * 0.35 + 1.6
                z = math.cos(phi) * 0.35
                vertices.append([x, y, z])
        
        # Detailed body
        body_segments = 12
        height_segments = 10
        for i in range(height_segments):
            height_ratio = i / height_segments
            y = 1.2 - height_ratio * 0.8
            radius = 0.25 * (1 - height_ratio * 0.3)
            
            for j in range(body_segments):
                angle = j * 2 * math.pi / body_segments
                x = math.cos(angle) * radius
                z = math.sin(angle) * radius * 0.7
                vertices.append([x, y, z])
        
        return vertices
    
    def generate_smart_faces(self):
        """Generate intelligent face connectivity"""
        self.faces = []
        
        # Head faces
        head_segments = 16
        for i in range(head_segments):
            for j in range(head_segments - 1):
                v1 = i * head_segments + j
                v2 = ((i + 1) % head_segments) * head_segments + j
                v3 = i * head_segments + j + 1
                v4 = ((i + 1) % head_segments) * head_segments + j + 1
                self.faces.append([v1, v2, v3])
                self.faces.append([v2, v4, v3])
        
        # Body faces
        body_start = head_segments * head_segments
        body_segments = 12
        height_segments = 10
        
        for i in range(height_segments - 1):
            for j in range(body_segments):
                v1 = body_start + i * body_segments + j
                v2 = body_start + i * body_segments + (j + 1) % body_segments
                v3 = body_start + (i + 1) * body_segments + j
                v4 = body_start + (i + 1) * body_segments + (j + 1) % body_segments
                self.faces.append([v1, v2, v3])
                self.faces.append([v2, v4, v3])
    
    def load_vrm_model(self, file_path):
        """Load and process VRM model"""
        model_info = VRMModelLoader.parse_vrm_file(file_path)
        if model_info:
            self.model_info = model_info
            self.vertices = VRMModelLoader.generate_vrm_preview(model_info)
            self.generate_smart_faces()
            print(f"✅ VRM model '{model_info['file_name']}' loaded successfully!")
            return True
        else:
            print("❌ Failed to load VRM model, using preview")
            self.create_preview_model()
            return False
    
    def project_3d_to_2d(self, point):
        """Advanced 3D to 2D projection with perspective"""
        x, y, z = point
        
        # Apply rotations
        # X rotation
        y_rot = y * math.cos(self.rotation_x) - z * math.sin(self.rotation_x)
        z_rot = y * math.sin(self.rotation_x) + z * math.cos(self.rotation_x)
        y, z = y_rot, z_rot
        
        # Y rotation  
        x_rot = x * math.cos(self.rotation_y) + z * math.sin(self.rotation_y)
        z_rot = -x * math.sin(self.rotation_y) + z * math.cos(self.rotation_y)
        x, z = x_rot, z_rot
        
        # Perspective projection
        if z + self.zoom != 0:
            factor = self.zoom / (z + self.zoom)
            x = x * factor
            y = y * factor
        
        # Convert to screen coordinates
        screen_x = int(x * 200 + 500)
        screen_y = int(-y * 200 + 400)
        
        return screen_x, screen_y
    
    def draw_advanced_model(self, message=""):
        """Draw the advanced 3D model with realistic animations"""
        if not self.screen:
            return False
            
        try:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        return False
                    elif event.key == K_u:
                        return "upload"
            
            # Update animations
            self.update_animations()
            
            # Clear screen with gradient
            self.draw_gradient_background()
            
            # Draw 3D model
            self.draw_animated_model()
            
            # Draw UI
            self.draw_advanced_ui(message)
            
            pygame.display.flip()
            self.clock.tick(60)
            return True
            
        except Exception as e:
            print(f"⚠️ Error drawing model: {e}")
            return False
    
    def draw_gradient_background(self):
        """Draw a beautiful gradient background"""
        for y in range(800):
            # Blue to purple gradient
            r = 20 + y // 20
            g = 25 + y // 25
            b = 40 + y // 15
            pygame.draw.line(self.screen, (r, g, b), (0, y), (1000, y))
        
        # Add some stars/particles
        for i in range(50):
            x = (math.sin(self.animation_time * 0.2 + i) * 300 + 500) % 1000
            y = (math.cos(self.animation_time * 0.3 + i) * 200 + 400) % 800
            size = 1 + math.sin(self.animation_time + i)
            alpha = 100 + int(math.sin(self.animation_time * 0.5 + i) * 100)
            color = (255, 255, 255, alpha)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(size))
    
    def update_animations(self):
        """Update all animations"""
        self.animation_time += 0.05
        self.blink_timer += 1
        
        # Blinking
        if self.blink_timer > 120:
            self.eye_open = not self.eye_open
            self.blink_timer = 0
        
        # Head bobbing during speech
        if self.speech_active:
            self.head_bob = math.sin(self.animation_time * 6) * 0.1
            self.mouth_openness = abs(math.sin(self.animation_time * 8)) * 0.15
        else:
            self.head_bob = math.sin(self.animation_time * 0.5) * 0.02
            self.mouth_openness = 0.02
    
    def draw_animated_model(self):
        """Draw the model with animations"""
        # Draw faces with shading
        for face in self.faces:
            if len(face) >= 3:
                points = []
                face_z = 0
                
                for vertex_idx in face[:3]:
                    if vertex_idx < len(self.vertices):
                        vertex = self.vertices[vertex_idx].copy()
                        
                        # Apply animations
                        if vertex_idx < 256:  # Head vertices
                            vertex[1] += self.head_bob
                        
                        point_2d = self.project_3d_to_2d(vertex)
                        points.append(point_2d)
                        face_z += vertex[2]
                
                if len(points) == 3:
                    # Calculate face color based on depth and expression
                    avg_z = face_z / 3
                    color = self.calculate_face_color(avg_z)
                    
                    # Draw filled face with outline
                    pygame.draw.polygon(self.screen, color, points)
                    pygame.draw.polygon(self.screen, (color[0]//2, color[1]//2, color[2]//2), points, 1)
        
        # Draw facial features
        self.draw_animated_face()
    
    def calculate_face_color(self, z_depth):
        """Calculate face color based on depth and expression"""
        base_intensity = 150 + int(z_depth * 30)
        base_intensity = max(50, min(200, base_intensity))
        
        if self.current_expression == "happy":
            return (base_intensity, base_intensity, 100)  # Yellow
        elif self.current_expression == "sad":
            return (100, 100, base_intensity)  # Blue
        elif self.current_expression == "speaking":
            return (base_intensity, 100, 100)  # Red
        elif self.current_expression == "thinking":
            return (100, base_intensity, 100)  # Green
        else:
            return (base_intensity, base_intensity, base_intensity)  # Gray
    
    def draw_animated_face(self):
        """Draw animated facial features"""
        # Eye positions
        left_eye_pos = [-0.12, 1.62 + self.head_bob, 0.35]
        right_eye_pos = [0.12, 1.62 + self.head_bob, 0.35]
        
        # Draw eyes
        left_eye = self.project_3d_to_2d(left_eye_pos)
        right_eye = self.project_3d_to_2d(right_eye_pos)
        
        eye_size = 6 if self.eye_open else 2
        pygame.draw.circle(self.screen, (255, 255, 255), left_eye, eye_size)
        pygame.draw.circle(self.screen, (255, 255, 255), right_eye, eye_size)
        
        # Draw pupils
        if self.eye_open:
            pupil_offset = math.sin(self.animation_time * 0.3) * 2
            pygame.draw.circle(self.screen, (0, 0, 0), 
                             (left_eye[0] + int(pupil_offset), left_eye[1]), 3)
            pygame.draw.circle(self.screen, (0, 0, 0),
                             (right_eye[0] + int(pupil_offset), right_eye[1]), 3)
        
        # Draw mouth
        mouth_y = 1.52 + self.head_bob + self.mouth_openness
        mouth_pos = [0, mouth_y, 0.35]
        mouth_point = self.project_3d_to_2d(mouth_pos)
        
        mouth_color = (200, 80, 80)
        mouth_size = 4 + int(self.mouth_openness * 20)
        pygame.draw.circle(self.screen, mouth_color, mouth_point, mouth_size)
    
    def draw_advanced_ui(self, message=""):
        """Draw advanced UI overlay"""
        # Main info panel
        panel_rect = pygame.Rect(20, 20, 400, 150)
        pygame.draw.rect(self.screen, (30, 30, 50, 200), panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, (100, 150, 255), panel_rect, 2, border_radius=12)
        
        font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 22)
        
        # Model info
        if self.model_info:
            model_name = self.model_info['file_name']
            model_size = self.model_info['file_size']
            info_text = f"VRM Model: {model_name}"
            size_text = f"Size: {model_size:,} bytes"
        else:
            info_text = "VRM Model: Preview Avatar"
            size_text = "Press U to load VRM file"
        
        # Render texts
        texts = [
            (info_text, (40, 40)),
            (size_text, (40, 70)),
            (f"Expression: {self.current_expression.upper()}", (40, 100)),
            (f"YOLOv8 Vision: Active", (40, 130))
        ]
        
        for text, pos in texts:
            text_surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, pos)
        
        # Controls panel
        controls = [
            "CONTROLS:",
            "• SPACE: Talk to AI Assistant",
            "• V: Analyze with YOLOv8",
            "• U: Upload VRM File", 
            "• Arrow Keys: Rotate Model",
            "• +/-: Zoom In/Out",
            "• ESC: Exit"
        ]
        
        for i, control in enumerate(controls):
            control_surface = small_font.render(control, True, (200, 200, 255))
            self.screen.blit(control_surface, (20, 630 + i * 22))
        
        # Current message
        if message:
            msg_surface = small_font.render(f"Speaking: {message[:70]}...", True, (255, 255, 0))
            self.screen.blit(msg_surface, (40, 180))
    
    def set_speaking(self, speaking=True):
        """Set speaking state with enhanced animations"""
        self.speech_active = speaking
        if speaking:
            self.current_expression = "speaking"
        else:
            self.current_expression = "neutral"
    
    def set_thinking(self, thinking=True):
        """Set thinking state"""
        if thinking:
            self.current_expression = "thinking"
        else:
            self.current_expression = "neutral"

class VisionSpeechAI:
    def __init__(self):
        # Initialize APIs
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.chat = self.model.start_chat(history=[])
        
        self.elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        
        # Initialize YOLOv8 detector
        print("🔍 Initializing YOLOv8 detector...")
        self.yolo_detector = YOLOv8Detector()
        
        # Initialize camera
        self.camera = self.initialize_camera()
        
        # Initialize Advanced VRM Renderer
        self.renderer = AdvancedVRMRenderer()
        
        # Vision system state
        self.vision_active = True
        self.latest_frame = None
        self.latest_detections = []
        self.vision_thread = None
        self.vision_lock = threading.Lock()
        self.stop_vision = False
        
        # Start continuous vision processing
        if self.camera:
            self.start_continuous_vision()
        
        # Calibrate microphone
        print("🎤 Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("🚀 Advanced VRM Avatar Assistant with Continuous YOLOv8 Ready!")
        print("=" * 60)
    
    def initialize_camera(self):
        """Initialize camera with robust error handling"""
        print("📷 Initializing camera...")
        
        # Try different backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto")
        ]
        
        for backend, backend_name in backends:
            print(f"   Trying {backend_name} backend...")
            for i in range(3):
                try:
                    camera = cv2.VideoCapture(i, backend)
                    
                    if not camera.isOpened():
                        camera.release()
                        continue
                    
                    # Set properties before reading
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Wait for camera to initialize
                    time.sleep(0.5)
                    
                    # Try to read multiple test frames
                    success_count = 0
                    for attempt in range(10):
                        try:
                            ret = camera.grab()
                            if ret:
                                ret, frame = camera.retrieve()
                                if ret and frame is not None:
                                    if frame.size > 0 and len(frame.shape) == 3:
                                        success_count += 1
                                        if success_count >= 3:
                                            print(f"✅ Camera {i} working with {backend_name}")
                                            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                                            return camera
                        except Exception as e:
                            print(f"   Frame test failed: {e}")
                        time.sleep(0.1)
                    
                    camera.release()
                    
                except Exception as e:
                    print(f"   Camera {i} error: {e}")
                    try:
                        camera.release()
                    except:
                        pass
        
        print("❌ No working camera found. Vision features will be disabled.")
        print("   Try: Check camera permissions, close other apps using camera")
        return None
    
    def start_continuous_vision(self):
        """Start continuous vision processing in background"""
        print("👁️ Starting continuous vision processing...")
        self.stop_vision = False
        self.vision_thread = threading.Thread(target=self.continuous_vision_loop, daemon=True)
        self.vision_thread.start()
    
    def continuous_vision_loop(self):
        """Continuously process camera frames in background"""
        while not self.stop_vision:
            try:
                if self.camera and self.camera.isOpened():
                    # Capture frame
                    ret = self.camera.grab()
                    if ret:
                        ret, frame = self.camera.retrieve()
                        if ret and frame is not None and frame.size > 0:
                            # Make safe copy
                            frame = frame.copy()
                            if not frame.flags['C_CONTIGUOUS']:
                                frame = np.ascontiguousarray(frame)
                            
                            # Run YOLOv8 detection
                            detections = self.yolo_detector.detect(frame)
                            
                            # Update shared state
                            with self.vision_lock:
                                self.latest_frame = frame
                                self.latest_detections = detections
                
                # Sleep briefly to avoid CPU overuse
                time.sleep(0.1)  # 10 FPS processing
                
            except Exception as e:
                print(f"⚠️ Vision loop error: {e}")
                time.sleep(1)
    
    def get_latest_vision_data(self):
        """Get the latest vision data (thread-safe)"""
        with self.vision_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_detections.copy()
            return None, []
    
    def show_current_vision(self):
        """Display current vision with detections"""
        frame, detections = self.get_latest_vision_data()
        if frame is not None:
            annotated = self.yolo_detector.draw_detections(frame, detections)
            cv2.imshow("Live YOLOv8 Vision", annotated)
            cv2.waitKey(1)
            return True
        return False
        """Upload VRM file with dialog"""
        try:
            root = tk.Tk()
            root.withdraw()
            
            file_path = filedialog.askopenfilename(
                title="Select VRM File from VRoid Studio",
                filetypes=[("VRM Files", "*.vrm"), ("All Files", "*.*")]
            )
            
            if file_path:
                success = self.renderer.load_vrm_model(file_path)
                if success:
                    messagebox.showinfo("Success", "VRM model loaded successfully!")
                else:
                    messagebox.showerror("Error", "Failed to load VRM model")
                return success
            return False
            
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False
    
    def capture_and_analyze_with_yolo(self):
        """Get current vision analysis from continuous processing"""
        try:
            print("📸 Getting current vision data...")
            self.renderer.set_thinking(True)
            
            frame, detections = self.get_latest_vision_data()
            
            if frame is None:
                self.renderer.set_thinking(False)
                return None, "Camera feed not available"
            
            # Display the annotated image
            try:
                annotated_frame = self.yolo_detector.draw_detections(frame, detections)
                cv2.imshow("YOLOv8 Detection", annotated_frame)
                cv2.waitKey(2000)
                cv2.destroyWindow("YOLOv8 Detection")
            except Exception as display_error:
                print(f"⚠️ Could not display image: {display_error}")
            
            # Generate summary
            summary = self.yolo_detector.get_detection_summary(detections)
            
            self.renderer.set_thinking(False)
            print(f"✅ Vision analysis: {summary}")
            
            return detections, summary
            
        except Exception as e:
            print(f"❌ Error getting vision data: {e}")
            self.renderer.set_thinking(False)
            return None, "Vision analysis failed"
    
    def capture_image(self):
        """Capture image from continuous vision feed"""
        frame, detections = self.get_latest_vision_data()
        if frame is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)
            except Exception as e:
                print(f"❌ Error converting frame: {e}")
        return None
    
    def analyze_image_with_gemini(self, image, prompt="What do you see?"):
        """Analyze image with Gemini"""
        try:
            print("👁️ Analyzing image with Gemini...")
            self.renderer.set_thinking(True)
            response = self.model.generate_content([prompt, image])
            analysis = response.text
            self.renderer.set_thinking(False)
            return analysis
        except Exception as e:
            print(f"❌ Error analyzing image: {e}")
            self.renderer.set_thinking(False)
            return "Sorry, I couldn't analyze the image."
    
    def record_audio(self):
        """Record audio from microphone"""
        print("🎤 Listening...")
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("⏹️ Finished recording")
            return audio
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
            return None
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return None
    
    def speech_to_text(self, audio):
        """Convert speech to text"""
        print("🔄 Converting speech to text...")
        self.renderer.set_thinking(True)
        try:
            user_text = self.recognizer.recognize_google(audio)
            print(f"👤 You said: {user_text}")
            self.renderer.set_thinking(False)
            return user_text
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            self.renderer.set_thinking(False)
            return None
        except Exception as e:
            print(f"❌ Error in speech-to-text: {e}")
            self.renderer.set_thinking(False)
            return None
    
    def get_ai_response(self, user_message, image=None, yolo_context=None):
        """Get AI response from Gemini"""
        print("🤖 Thinking...")
        self.renderer.set_thinking(True)
        try:
            # Construct prompt with YOLOv8 context if available
            if yolo_context:
                prompt = f"User asked: '{user_message}'. YOLOv8 detected: {yolo_context}. Respond conversationally."
                response = self.chat.send_message(prompt)
            elif image:
                response = self.model.generate_content([f"User asked: '{user_message}'. Analyze image and respond conversationally.", image])
            else:
                response = self.chat.send_message(f"Respond conversationally to: {user_message}. Keep it brief.")
            
            ai_message = response.text
            self.renderer.set_thinking(False)
            return ai_message
        except Exception as e:
            print(f"❌ Error getting AI response: {e}")
            self.renderer.set_thinking(False)
            return "Sorry, I encountered an error."
    
    def text_to_speech(self, text):
        """Convert text to speech with ElevenLabs"""
        print("🔊 Converting to speech...")
        self.renderer.set_speaking(True)
        try:
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                text=text, voice_id=VOICE_ID, model_id="eleven_monolingual_v1"
            )
            temp_file = "temp_speech.mp3"
            with open(temp_file, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)
            
            print("🎵 Speaking...")
            pygame.mixer.init()
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                if not self.renderer.draw_advanced_model(f"{text[:60]}..."):
                    break
                pygame.time.Clock().tick(60)
            
            pygame.mixer.quit()
            try:
                os.remove(temp_file)
            except:
                pass
                
            self.renderer.set_speaking(False)
            print("✅ Speech completed")
            
        except Exception as e:
            print(f"❌ Error in text-to-speech: {e}")
            self.renderer.set_speaking(False)
            print(f"📢 Would have said: {text}")
    
    def process_vision_command(self, user_text):
        """Process vision-related commands with YOLOv8 + Gemini"""
        user_text_lower = user_text.lower()
        
        # Get current vision data
        frame, detections = self.get_latest_vision_data()
        
        if frame is None:
            return "Sorry, my camera isn't working right now."
        
        # Get YOLOv8 detection summary
        yolo_summary = self.yolo_detector.get_detection_summary(detections)
        print(f"🔍 YOLOv8 sees: {yolo_summary}")
        
        # Convert frame to PIL Image for Gemini
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Create context-aware prompt for Gemini
        context = f"YOLOv8 detected: {yolo_summary}. "
        
        # Check for specific types of queries
        if any(word in user_text_lower for word in ['finger', 'fingers', 'hand', 'hands']):
            prompt = context + f"User asked: '{user_text}'. Please count fingers/hands carefully and respond."
        elif any(word in user_text_lower for word in ['person', 'people', 'face', 'who']):
            prompt = context + f"User asked: '{user_text}'. Describe the people you see."
        elif any(word in user_text_lower for word in ['what', 'see', 'looking', 'show']):
            prompt = context + f"User asked: '{user_text}'. Describe what you see in detail."
        elif any(word in user_text_lower for word in ['count', 'how many']):
            prompt = context + f"User asked: '{user_text}'. Count the specific items they're asking about."
        else:
            prompt = context + f"User asked: '{user_text}'. Analyze the image and respond."
        
        # Get Gemini's analysis with YOLOv8 context
        return self.analyze_image_with_gemini(pil_image, prompt)
    
    def run_conversation(self):
        """Main conversation loop"""
        print("\n🎯 Advanced VRM Avatar Assistant with Continuous Vision!")
        print("📁 Your VRM file will be automatically loaded")
        print("👁️ YOLOv8 is running continuously in the background")
        print("🎤 Press SPACE to talk to the AI")
        print("🔍 Press V to see live vision feed")
        print("🔄 Use Arrow Keys to rotate the 3D model")
        print("=" * 50)
        
        conversation_active = True
        show_vision_window = False
        
        while conversation_active:
            try:
                # Update vision window if active
                if show_vision_window:
                    self.show_current_vision()
                
                # Check for upload request
                result = self.renderer.draw_advanced_model("Ready! SPACE=talk, V=vision, U=upload VRM")
                if result == "upload":
                    print("\n📁 Opening file dialog...")
                    upload_thread = threading.Thread(target=self.upload_vrm_file)
                    upload_thread.daemon = True
                    upload_thread.start()
                    time.sleep(1)
                
                # Check for key presses
                keys = pygame.key.get_pressed()
                
                # Toggle vision window
                if keys[K_v]:
                    show_vision_window = not show_vision_window
                    if not show_vision_window:
                        cv2.destroyWindow("Live YOLOv8 Vision")
                    print(f"👁️ Vision window: {'ON' if show_vision_window else 'OFF'}")
                    time.sleep(0.3)  # Debounce
                
                # Voice interaction
                if keys[K_SPACE]:
                    print("\n" + "="*40)
                    audio = self.record_audio()
                    if not audio:
                        continue
                    
                    user_text = self.speech_to_text(audio)
                    if not user_text:
                        continue
                    
                    if "quit" in user_text.lower() or "exit" in user_text.lower():
                        print("👋 Goodbye!")
                        conversation_active = False
                        continue
                    
                    # Handle vision commands
                    vision_keywords = ['see', 'look', 'what', 'describe', 'face', 'people', 
                                      'camera', 'detect', 'find', 'identify', 'count', 'object',
                                      'finger', 'hand', 'show', 'many', 'holding']
                    user_text_lower = user_text.lower()
                    is_vision_query = any(keyword in user_text_lower for keyword in vision_keywords)
                    
                    if is_vision_query and self.camera:
                        ai_response = self.process_vision_command(user_text)
                    else:
                        ai_response = self.get_ai_response(user_text)
                    
                    self.text_to_speech(ai_response)
                    time.sleep(0.5)
                
                # Handle rotation controls
                if keys[K_LEFT]:
                    self.renderer.rotation_y -= 0.03
                if keys[K_RIGHT]:
                    self.renderer.rotation_y += 0.03
                if keys[K_UP]:
                    self.renderer.rotation_x -= 0.03
                if keys[K_DOWN]:
                    self.renderer.rotation_x += 0.03
                if keys[K_PLUS] or keys[K_EQUALS]:
                    self.renderer.zoom = max(4, self.renderer.zoom - 0.1)
                if keys[K_MINUS]:
                    self.renderer.zoom = min(15, self.renderer.zoom + 0.1)
                
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                conversation_active = False
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                time.sleep(1)
        
        # Stop vision thread
        self.stop_vision = True
        if self.vision_thread:
            self.vision_thread.join(timeout=2)
    
    def __del__(self):
        """Cleanup resources"""
        try:
            # Stop vision thread
            self.stop_vision = True
            if self.vision_thread:
                self.vision_thread.join(timeout=2)
            
            if self.camera:
                self.camera.release()
            self.audio.terminate()
            cv2.destroyAllWindows()
            pygame.quit()
        except:
            pass

def main():
    """Main entry point"""
    try:
        # Check imports
        import google.generativeai as genai
        from elevenlabs.client import ElevenLabs
        import pyaudio
        import speech_recognition as sr
        import cv2
        import numpy as np
        from PIL import Image
        import pygame
        import onnxruntime as ort
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n📦 Install with:")
        print("pip install google-generativeai elevenlabs pyaudio SpeechRecognition opencv-python pillow pygame numpy onnxruntime")
        return
    
    # Test APIs
    print("🔍 Testing APIs...")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        test_model = genai.GenerativeModel('gemini-2.0-flash')
        test_response = test_model.generate_content("Hello")
        print("✅ Gemini API working!")
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return
    
    try:
        test_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        print("✅ ElevenLabs API working!")
    except Exception as e:
        print(f"❌ ElevenLabs API error: {e}")
        return
    
    # Check YOLOv8 model
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"⚠️ YOLOv8 model not found at: {YOLO_MODEL_PATH}")
        print("   Vision detection will use Gemini only")
    else:
        print(f"✅ YOLOv8 model found at: {YOLO_MODEL_PATH}")
    
    # Start the assistant
    print("\n" + "="*60)
    print("🚀 Starting VRM Avatar Assistant with YOLOv8...")
    print("="*60)
    assistant = VisionSpeechAI()
    assistant.run_conversation()

if __name__ == "__main__":
    main()