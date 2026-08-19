from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from matcher import HighAccuracyMatcher


class ExactFindAI:
    def __init__(
        self,
        root: tk.Tk,
        camera_index: int = 0,
        target_model_name: str = "yoloe-26s-seg.pt",
        target_confidence: float = 0.04,
        scene_model_name: str = "yolo11s.pt",
        scene_confidence: float = 0.22,
        image_size: int = 640,
        verify_threshold: float = 0.72,
        dino_min: float = 0.70,
        ambiguity_margin: float = 0.055,
        confirm_frames: int = 3,
        dino_model: str = "dinov2_vitb14_reg",
    ):
        self.root = root
        self.root.title("ExactFind AI - High Accuracy")
        self.root.geometry("1240x900")

        self.camera_index = camera_index
        self.target_confidence = target_confidence
        self.scene_confidence = scene_confidence
        self.image_size = image_size

        self.verify_threshold = verify_threshold
        self.dino_min = dino_min
        self.ambiguity_margin = ambiguity_margin
        self.confirm_frames = max(1, confirm_frames)

        self.running = True
        self.mode = "capture"
        self.photo = None
        self.reference_photo = None
        self.latest_frame = None

        self.reference_crops = []
        self.reference_scene_names = []
        self.reference_main_crop = None

        self.frame_counter = 0
        self.last_scene_detections = []
        self.last_target_box = None
        self.confirmation_hits = 0
        self.miss_frames = 0
        self.confirmed_target = None

        self.output_dir = Path("data/reference")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading scene model: {scene_model_name}")
        self.scene_model = YOLO(scene_model_name)

        print(f"Loading target model: {target_model_name}")
        self.target_model = YOLOE(target_model_name)

        # DINOv2 + SIFT verifier.
        self.matcher = HighAccuracyMatcher(dino_model_name=dino_model)

        print(f"Opening webcam {camera_index}...")
        self.cap = self._open_camera(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(20, self.update_frame)

    @staticmethod
    def _open_camera(index):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
        return cv2.VideoCapture(index)

    def _build_ui(self):
        tk.Label(
            self.root,
            text="ExactFind AI",
            font=("Arial", 23, "bold"),
        ).pack(pady=(10, 1))

        tk.Label(
            self.root,
            text="High-accuracy exact object search from an uploaded or captured reference",
            font=("Arial", 12),
        ).pack(pady=(0, 7))

        self.status = tk.Label(
            self.root,
            text="STEP 1: Capture a target or upload target image(s)",
            font=("Arial", 14, "bold"),
        )
        self.status.pack(pady=(2, 8))

        body = tk.Frame(self.root)
        body.pack(expand=True, fill="both", padx=15)

        left = tk.Frame(body)
        left.pack(side="left", expand=True, fill="both", padx=(0, 10))

        tk.Label(left, text="LIVE CAMERA", font=("Arial", 11, "bold")).pack()
        self.video_label = tk.Label(left)
        self.video_label.pack(expand=True)

        side = tk.Frame(body, width=290)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)

        tk.Label(side, text="SELECTED TARGET", font=("Arial", 11, "bold")).pack(pady=(0, 8))

        self.reference_label = tk.Label(
            side,
            text="No target selected yet",
            width=31,
            height=13,
            relief="groove",
        )
        self.reference_label.pack(pady=(0, 10))

        self.view_label = tk.Label(
            side,
            text="Reference views: 0",
            font=("Arial", 10, "bold"),
            anchor="w",
        )
        self.view_label.pack(fill="x")

        self.score_label = tk.Label(
            side,
            text="Match score: --",
            font=("Arial", 10),
            justify="left",
            anchor="w",
        )
        self.score_label.pack(fill="x", pady=8)

        tk.Label(
            side,
            text=(
                "Green = normal YOLO object\n"
                "Red = confirmed exact target\n"
                "Yellow = capture region\n\n"
                "For best accuracy, add 2-4 views\n"
                "of the SAME physical object."
            ),
            font=("Arial", 10),
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=4)

        controls = tk.Frame(self.root)
        controls.pack(pady=10)

        self.capture_button = tk.Button(
            controls,
            text="1. Capture Target",
            command=self.capture_target,
            width=17,
            height=2,
        )
        self.capture_button.grid(row=0, column=0, padx=5)

        self.upload_button = tk.Button(
            controls,
            text="Upload Target",
            command=self.upload_target,
            width=17,
            height=2,
        )
        self.upload_button.grid(row=0, column=1, padx=5)

        self.search_button = tk.Button(
            controls,
            text="2. Start Search",
            command=self.start_search,
            width=17,
            height=2,
            state="disabled",
        )
        self.search_button.grid(row=0, column=2, padx=5)

        self.reset_button = tk.Button(
            controls,
            text="Reset",
            command=self.reset_target,
            width=13,
            height=2,
        )
        self.reset_button.grid(row=0, column=3, padx=5)

        tk.Button(
            controls,
            text="Close",
            command=self.close,
            width=13,
            height=2,
        ).grid(row=0, column=4, padx=5)

        tk.Label(
            self.root,
            text=(
                "Capture: keep one target near the center. After the first capture, "
                "you may rotate it and click Add Target View before Start Search."
            ),
            font=("Arial", 10),
        ).pack(pady=(0, 8))

    @staticmethod
    def get_capture_roi(frame):
        h, w = frame.shape[:2]
        roi_w = int(w * 0.52)
        roi_h = int(h * 0.62)
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        return x1, y1, x1 + roi_w, y1 + roi_h

    @staticmethod
    def box_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def center_inside(box, region):
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = region
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def detect_scene_objects(self, frame, conf=None):
        result = self.scene_model.predict(
            source=frame,
            conf=self.scene_confidence if conf is None else conf,
            imgsz=self.image_size,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().tolist())
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "name": result.names[cls_id],
                    "score": score,
                }
            )
        return detections

    def _tight_reference_crop(self, image, preferred_region=None):
        h, w = image.shape[:2]
        fallback = preferred_region or (0, 0, w, h)

        try:
            detections = self.detect_scene_objects(image, conf=0.16)
        except Exception as exc:
            print("Auto-crop detector error:", exc)
            detections = []

        candidates = []
        fx1, fy1, fx2, fy2 = fallback
        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0

        for det in detections:
            if preferred_region is not None and not self.center_inside(det["box"], preferred_region):
                continue

            x1, y1, x2, y2 = det["box"]
            area = max(1, (x2 - x1) * (y2 - y1))
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
            candidates.append({**det, "area": area, "dist": dist})

        if preferred_region is None and candidates:
            chosen = max(candidates, key=lambda d: d["area"])
        elif candidates:
            candidates.sort(key=lambda d: (d["dist"], -d["area"]))
            chosen = candidates[0]
        else:
            chosen = None

        if chosen is not None:
            x1, y1, x2, y2 = chosen["box"]
            pad_x = int((x2 - x1) * 0.04)
            pad_y = int((y2 - y1) * 0.04)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = image[y1:y2, x1:x2].copy()
            return crop, chosen["name"]

        x1, y1, x2, y2 = fallback
        return image[y1:y2, x1:x2].copy(), None

    def _add_reference(self, crop, scene_name, source_name):
        if crop is None or crop.size == 0:
            messagebox.showerror("ExactFind AI", "Could not create target crop.")
            return

        # Do not silently accumulate too many views and make enrollment slow.
        if len(self.reference_crops) >= 5:
            messagebox.showinfo(
                "ExactFind AI",
                "Five reference views are enough. Click Start Search.",
            )
            return

        self.matcher.add_reference(crop, scene_name=scene_name)
        self.reference_crops.append(crop.copy())
        self.reference_scene_names.append(scene_name)
        self.reference_main_crop = self.reference_crops[0]

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = self.output_dir / f"{source_name}_{len(self.reference_crops)}_{timestamp}.jpg"
        cv2.imwrite(str(save_path), crop)

        self._show_reference(self.reference_main_crop)
        self.mode = "ready"
        self.search_button.config(state="normal")
        self.capture_button.config(text="Add Target View")

        self.view_label.config(text=f"Reference views: {len(self.reference_crops)}")
        self.status.config(
            text=(
                f"TARGET READY ({len(self.reference_crops)} view"
                f"{'s' if len(self.reference_crops) != 1 else ''}) - "
                "add another view or click Start Search"
            )
        )

    def capture_target(self):
        if self.latest_frame is None:
            return
        frame = self.latest_frame.copy()
        roi = self.get_capture_roi(frame)
        crop, name = self._tight_reference_crop(frame, preferred_region=roi)
        self._add_reference(crop, name, "captured_target")

    def upload_target(self):
        paths = filedialog.askopenfilenames(
            title="Select 1-5 images of the SAME target object",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        for path in list(paths)[:5]:
            image = cv2.imread(path)
            if image is None:
                continue
            crop, name = self._tight_reference_crop(image, preferred_region=None)
            self._add_reference(crop, name, "uploaded_target")

    def _show_reference(self, crop):
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((255, 280))
        self.reference_photo = ImageTk.PhotoImage(image=image)
        self.reference_label.config(
            image=self.reference_photo,
            text="",
            width=255,
            height=280,
        )

    def _dominant_reference_class(self):
        names = [x for x in self.reference_scene_names if x]
        if not names:
            return None
        return Counter(names).most_common(1)[0][0]

    def start_search(self):
        if not self.reference_crops:
            self.status.config(text="Capture or upload a target first")
            return

        self.status.config(text="Creating target identity...")
        self.root.update_idletasks()

        # Build YOLOE's visual prompt from the first real reference view.
        ref = self.reference_main_crop
        h, w = ref.shape[:2]
        prompts = {
            "bboxes": np.array([[0, 0, w - 1, h - 1]], dtype=np.float32),
            "cls": np.array([0], dtype=np.int32),
        }

        try:
            self.target_model.predict(
                source=ref,
                refer_image=ref,
                visual_prompts=prompts,
                predictor=YOLOEVPSegPredictor,
                conf=0.01,
                imgsz=self.image_size,
                verbose=False,
            )
        except Exception as exc:
            self.status.config(text="Could not create YOLOE visual prompt")
            self.score_label.config(text=str(exc))
            return

        self.mode = "search"
        self.confirmation_hits = 0
        self.miss_frames = 0
        self.last_target_box = None
        self.confirmed_target = None
        self.frame_counter = 0

        self.capture_button.config(state="disabled")
        self.upload_button.config(state="disabled")
        self.search_button.config(state="disabled")

        self.status.config(text="SEARCHING FOR THE SELECTED TARGET...")
        self.score_label.config(text="Match score: searching...")

    def reset_target(self):
        self.mode = "capture"
        self.reference_crops.clear()
        self.reference_scene_names.clear()
        self.reference_main_crop = None
        self.matcher.clear()

        self.confirmation_hits = 0
        self.miss_frames = 0
        self.last_target_box = None
        self.confirmed_target = None

        self.reference_label.config(
            image="",
            text="No target selected yet",
            width=31,
            height=13,
        )
        self.view_label.config(text="Reference views: 0")
        self.score_label.config(text="Match score: --")

        self.capture_button.config(text="1. Capture Target", state="normal")
        self.upload_button.config(state="normal")
        self.search_button.config(state="disabled")
        self.status.config(text="STEP 1: Capture a target or upload target image(s)")

    def detect_yoloe_candidates(self, frame):
        result = self.target_model.predict(
            source=frame,
            conf=self.target_confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]

        if result.boxes is None:
            return []

        h, w = frame.shape[:2]
        area_frame = float(h * w)
        polygons = None

        if result.masks is not None:
            try:
                polygons = result.masks.xy
            except Exception:
                polygons = None

        candidates = []
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].detach().cpu().tolist())
            score = float(box.conf[0].item())

            if polygons is not None and i < len(polygons):
                poly = np.asarray(polygons[i])
                if poly.ndim == 2 and len(poly) >= 3:
                    x1 = int(np.floor(poly[:, 0].min()))
                    y1 = int(np.floor(poly[:, 1].min()))
                    x2 = int(np.ceil(poly[:, 0].max()))
                    y2 = int(np.ceil(poly[:, 1].max()))

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / area_frame
            if area_ratio < 0.001 or area_ratio > 0.65:
                continue

            candidates.append(
                {
                    "box": (x1, y1, x2, y2),
                    "score": score,
                }
            )

        return candidates

    def yoloe_score_for_scene_box(self, scene_box, yoloe_candidates):
        best_iou = 0.0
        best_conf = 0.0

        for cand in yoloe_candidates:
            iou = self.box_iou(scene_box, cand["box"])
            if iou > best_iou:
                best_iou = iou
                best_conf = cand["score"]

        if best_iou < 0.12:
            return 0.0, best_iou

        # YOLOE prompt confidence tends to be lower than closed-set YOLO.
        component = float(np.clip(best_conf / 0.28, 0.0, 1.0))
        component *= float(np.clip(best_iou / 0.55, 0.0, 1.0))
        return component, best_iou

    def verify_all_scene_objects(self, frame, scene_detections, yoloe_candidates):
        if not scene_detections:
            return []

        dominant_class = self._dominant_reference_class()

        prepared = []
        crops = []

        for det in scene_detections:
            x1, y1, x2, y2 = det["box"]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            yoloe_component, yoloe_iou = self.yoloe_score_for_scene_box(
                det["box"], yoloe_candidates
            )

            if dominant_class is None:
                class_match = 0.5
            elif det["name"] == dominant_class:
                class_match = 1.0
            else:
                class_match = 0.15

            prepared.append(
                (det, crop, yoloe_component, yoloe_iou, class_match)
            )
            crops.append(crop)

        if not prepared:
            return []

        # One DINOv2 forward pass for all visible object crops.
        embeddings = self.matcher.embed_batch(crops)

        verified = []
        for i, (det, crop, yoloe_component, yoloe_iou, class_match) in enumerate(prepared):
            scores = self.matcher.score_candidate(
                crop,
                yoloe_component=yoloe_component,
                class_match=class_match,
                embedding=embeddings[i:i + 1],
            )

            verified.append(
                {
                    **det,
                    **scores,
                    "yoloe_iou": yoloe_iou,
                }
            )

        verified.sort(key=lambda d: d["score"], reverse=True)
        return verified

    def select_candidate(self, verified):
        if not verified:
            return None, "no candidates"

        best = verified[0]
        second_score = verified[1]["score"] if len(verified) > 1 else 0.0
        margin = best["score"] - second_score

        if best["dino"] < self.dino_min:
            return None, f"DINO too low ({best['dino']:.3f})"

        if best["score"] < self.verify_threshold:
            return None, f"score too low ({best['score']:.3f})"

        if len(verified) > 1 and margin < self.ambiguity_margin:
            return None, f"ambiguous margin ({margin:.3f})"

        # For textured references, require at least a little local evidence
        # unless DINO is exceptionally strong.
        if (
            not best["weak_reference"]
            and best["sift"] < 0.10
            and best["dino"] < 0.86
        ):
            return None, "weak geometric evidence"

        best["second_score"] = second_score
        best["margin"] = margin
        return best, "accepted"

    def temporal_confirm(self, candidate):
        if candidate is None:
            self.confirmation_hits = max(0, self.confirmation_hits - 1)
            self.miss_frames += 1

            # Keep a confirmed target for one short missed frame to reduce
            # flicker, but never indefinitely.
            if self.confirmed_target is not None and self.miss_frames <= 1:
                return self.confirmed_target

            if self.confirmation_hits == 0:
                self.last_target_box = None
                self.confirmed_target = None
            return None

        self.miss_frames = 0

        if self.last_target_box is None:
            self.confirmation_hits = 1
        else:
            iou = self.box_iou(self.last_target_box, candidate["box"])
            if iou >= 0.18:
                self.confirmation_hits += 1
            else:
                self.confirmation_hits = 1

        self.last_target_box = candidate["box"]

        if self.confirmation_hits >= self.confirm_frames:
            self.confirmed_target = candidate
            return candidate

        return None

    @staticmethod
    def draw_scene(frame, detections, target_box=None):
        green = (0, 205, 0)
        for det in detections:
            if target_box is not None and ExactFindAI.box_iou(det["box"], target_box) >= 0.35:
                continue

            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)
            cv2.putText(
                frame,
                f"{det['name']} {det['score']:.2f}",
                (x1, max(24, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                green,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def draw_target(frame, target):
        if target is None:
            return

        x1, y1, x2, y2 = target["box"]
        red = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), red, 4)
        cv2.putText(
            frame,
            f"TARGET  {target['score'] * 100:.1f}%",
            (x1, max(28, y1 - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            red,
            2,
            cv2.LINE_AA,
        )

    def update_frame(self):
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status.config(text="Could not read webcam frame")
            self.root.after(100, self.update_frame)
            return

        self.latest_frame = frame.copy()
        display = frame.copy()

        if self.mode in ("capture", "ready"):
            x1, y1, x2, y2 = self.get_capture_roi(display)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 215, 255), 3)
            cv2.putText(
                display,
                "PLACE ONE TARGET OBJECT INSIDE THIS BOX",
                (max(10, x1 - 30), max(30, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )

        elif self.mode == "search":
            self.frame_counter += 1

            try:
                # Draw all normal objects every frame.
                self.last_scene_detections = self.detect_scene_objects(frame)
                yoloe_candidates = self.detect_yoloe_candidates(frame)

                verified = self.verify_all_scene_objects(
                    frame,
                    self.last_scene_detections,
                    yoloe_candidates,
                )

                candidate, reason = self.select_candidate(verified)
                target = self.temporal_confirm(candidate)

            except Exception as exc:
                print("Live inference error:", exc)
                self.status.config(text="LIVE INFERENCE ERROR")
                self.score_label.config(text=str(exc))
                self.root.after(250, self.update_frame)
                return

            target_box = target["box"] if target is not None else None
            self.draw_scene(display, self.last_scene_detections, target_box)
            self.draw_target(display, target)

            if target is not None:
                self.status.config(
                    text=f"TARGET FOUND - {target['score'] * 100:.1f}% match"
                )
                self.score_label.config(
                    text=(
                        f"Match score: {target['score'] * 100:.1f}%\n"
                        f"DINOv2: {target['dino']:.3f}\n"
                        f"SIFT: {target['sift']:.3f} "
                        f"({target['sift_inliers']} inliers)\n"
                        f"Color: {target['color']:.3f}\n"
                        f"YOLOE cue: {target['yoloe_component']:.3f}\n"
                        f"Margin: {target.get('margin', 0.0):.3f}\n"
                        f"Confirm: {self.confirmation_hits}/{self.confirm_frames}"
                    )
                )
            elif verified:
                best = verified[0]
                second = verified[1]["score"] if len(verified) > 1 else 0.0
                self.status.config(text="VERIFYING POSSIBLE TARGET...")
                self.score_label.config(
                    text=(
                        f"Best: {best['score'] * 100:.1f}%\n"
                        f"DINOv2: {best['dino']:.3f}\n"
                        f"SIFT: {best['sift']:.3f}\n"
                        f"2nd: {second * 100:.1f}%\n"
                        f"Decision: {reason}\n"
                        f"Confirm: {self.confirmation_hits}/{self.confirm_frames}"
                    )
                )
            else:
                self.status.config(text="SEARCHING FOR THE SELECTED TARGET...")
                self.score_label.config(
                    text=(
                        f"Required score: {self.verify_threshold:.2f}\n"
                        f"Required DINO: {self.dino_min:.2f}\n"
                        f"Scene objects: {len(self.last_scene_detections)}"
                    )
                )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((900, 660))
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.config(image=self.photo)

        self.root.after(10, self.update_frame)

    def close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target-model", default="yoloe-26s-seg.pt")
    parser.add_argument("--scene-model", default="yolo11s.pt")
    parser.add_argument("--target-conf", type=float, default=0.04)
    parser.add_argument("--scene-conf", type=float, default=0.22)
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--verify-threshold", type=float, default=0.72)
    parser.add_argument("--dino-min", type=float, default=0.70)
    parser.add_argument("--ambiguity-margin", type=float, default=0.055)
    parser.add_argument("--confirm-frames", type=int, default=3)

    parser.add_argument(
        "--dino-model",
        default="dinov2_vitb14_reg",
        choices=[
            "dinov2_vits14_reg",
            "dinov2_vitb14_reg",
            "dinov2_vitl14_reg",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()

    ExactFindAI(
        root=root,
        camera_index=args.source,
        target_model_name=args.target_model,
        target_confidence=args.target_conf,
        scene_model_name=args.scene_model,
        scene_confidence=args.scene_conf,
        image_size=args.imgsz,
        verify_threshold=args.verify_threshold,
        dino_min=args.dino_min,
        ambiguity_margin=args.ambiguity_margin,
        confirm_frames=args.confirm_frames,
        dino_model=args.dino_model,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
