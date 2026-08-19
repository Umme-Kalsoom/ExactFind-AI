from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


@dataclass
class ReferenceView:
    image: np.ndarray
    dino_embeddings: torch.Tensor
    hist: np.ndarray
    gray: np.ndarray
    keypoints: list
    descriptors: Optional[np.ndarray]
    aspect_ratio: float
    scene_name: Optional[str]


class HighAccuracyMatcher:
    """
    Exact-instance verifier.

    Main cues:
      1. DINOv2 ViT-B/14 with registers (global visual identity)
      2. SIFT + ratio test + RANSAC homography (fine local detail)
      3. HSV color histogram
      4. shape/aspect ratio consistency
      5. optional YOLOE visual-prompt overlap score

    The matcher supports several real reference views.  If the user gives only
    one view, mild photometric/geometric augmentations are embedded as a small
    reference bank to improve lighting and camera-angle tolerance.
    """

    def __init__(
        self,
        dino_model_name: str = "dinov2_vitb14_reg",
        device: Optional[str] = None,
        dino_size: int = 518,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dino_model_name = dino_model_name
        self.dino_size = dino_size

        print(f"Loading DINOv2 model: {dino_model_name} on {self.device}")
        print("First run can download the official DINOv2 code and weights.")
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            dino_model_name,
            trust_repo=True,
        )
        self.model.eval().to(self.device)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (dino_size, dino_size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self.sift = cv2.SIFT_create(
            nfeatures=1800,
            contrastThreshold=0.025,
            edgeThreshold=12,
            sigma=1.4,
        )
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),  # FLANN_INDEX_KDTREE
            dict(checks=64),
        )

        self.views: List[ReferenceView] = []

    @staticmethod
    def _letterbox_square(image: np.ndarray, pad_value: int = 127) -> np.ndarray:
        h, w = image.shape[:2]
        side = max(h, w)
        canvas = np.full((side, side, 3), pad_value, dtype=np.uint8)
        y = (side - h) // 2
        x = (side - w) // 2
        canvas[y:y + h, x:x + w] = image
        return canvas

    def _to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        square = self._letterbox_square(image_bgr)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.preprocess(pil).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def embed(self, image_bgr: np.ndarray) -> torch.Tensor:
        x = self._to_tensor(image_bgr)

        # Hub DINOv2 backbones return the class-token feature when called
        # directly.  Normalize it so cosine similarity is a dot product.
        feat = self.model(x)

        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            else:
                # Defensive fallback for a future compatible return format.
                feat = next(v for v in feat.values() if torch.is_tensor(v))

        if feat.ndim > 2:
            feat = feat.flatten(1)

        return F.normalize(feat.float(), dim=1).cpu()

    @torch.inference_mode()
    def embed_batch(self, images_bgr: List[np.ndarray]) -> torch.Tensor:
        """Embed all live candidate crops in one DINOv2 forward pass."""
        if not images_bgr:
            return torch.empty((0, 0), dtype=torch.float32)

        tensors = [self._to_tensor(img).squeeze(0) for img in images_bgr]
        x = torch.stack(tensors, dim=0).to(self.device)

        feat = self.model(x)

        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            else:
                feat = next(v for v in feat.values() if torch.is_tensor(v))

        if feat.ndim > 2:
            feat = feat.flatten(1)

        return F.normalize(feat.float(), dim=1).cpu()

    @staticmethod
    def _augmentations(image: np.ndarray) -> List[np.ndarray]:
        """
        Conservative virtual views.  We intentionally do NOT horizontally flip
        the target because logos/text/zipper layouts are useful exact-instance
        identity evidence.
        """
        out = [image.copy()]
        h, w = image.shape[:2]

        # Lighting variants.
        for alpha, beta in ((0.90, 5), (1.10, -5)):
            aug = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            out.append(aug)

        # Very small rotations preserve identity but improve hand-held camera
        # tolerance.
        center = (w / 2.0, h / 2.0)
        for angle in (-5.0, 5.0):
            mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug = cv2.warpAffine(
                image,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            out.append(aug)

        return out

    @staticmethod
    def _histogram(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [40, 40],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def add_reference(self, image: np.ndarray, scene_name: Optional[str] = None):
        if image is None or image.size == 0:
            raise ValueError("Empty reference image")

        embeddings = []
        for aug in self._augmentations(image):
            embeddings.append(self.embed(aug))

        dino_embeddings = torch.cat(embeddings, dim=0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)
        aspect_ratio = image.shape[1] / max(1.0, float(image.shape[0]))

        self.views.append(
            ReferenceView(
                image=image.copy(),
                dino_embeddings=dino_embeddings,
                hist=self._histogram(image),
                gray=gray,
                keypoints=kp or [],
                descriptors=desc,
                aspect_ratio=aspect_ratio,
                scene_name=scene_name,
            )
        )

        print(
            f"Added reference view #{len(self.views)} | "
            f"SIFT keypoints={len(kp or [])} | class={scene_name}"
        )

    def clear(self):
        self.views.clear()

    def reference_count(self) -> int:
        return len(self.views)

    @torch.inference_mode()
    def dino_similarity(
        self,
        crop: Optional[np.ndarray] = None,
        embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[float, int]:
        if not self.views:
            return 0.0, -1

        if embedding is None:
            if crop is None:
                raise ValueError("crop or embedding must be provided")
            q = self.embed(crop)
        else:
            q = embedding
            if q.ndim == 1:
                q = q.unsqueeze(0)
            q = F.normalize(q.float(), dim=1).cpu()

        best_score = -1.0
        best_view = -1

        for i, view in enumerate(self.views):
            sims = q @ view.dino_embeddings.T
            vals = torch.sort(sims.flatten(), descending=True).values

            if len(vals) >= 2:
                score = float((0.72 * vals[0] + 0.28 * vals[1]).item())
            else:
                score = float(vals[0].item())

            if score > best_score:
                best_score = score
                best_view = i

        return float(np.clip(best_score, -1.0, 1.0)), best_view

    def color_similarity(self, crop: np.ndarray, view_index: int) -> float:
        if view_index < 0 or view_index >= len(self.views):
            return 0.0

        hist = self._histogram(crop)
        corr = cv2.compareHist(
            self.views[view_index].hist,
            hist,
            cv2.HISTCMP_CORREL,
        )
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def shape_similarity(self, crop: np.ndarray, view_index: int) -> float:
        if view_index < 0 or view_index >= len(self.views):
            return 0.0

        h, w = crop.shape[:2]
        ratio = w / max(1.0, float(h))
        ref_ratio = self.views[view_index].aspect_ratio

        lo = min(ratio, ref_ratio)
        hi = max(ratio, ref_ratio)
        return float(np.clip(lo / max(hi, 1e-6), 0.0, 1.0))

    def sift_geometry(self, crop: np.ndarray, view_index: int) -> dict:
        """
        Returns a conservative local-detail score.

        For strongly textured objects, a valid homography/inlier ratio is
        powerful exact-instance evidence.  For textureless objects we expose a
        `weak_reference=True` flag so the caller can reduce this cue's weight
        instead of incorrectly rejecting the target.
        """
        if view_index < 0 or view_index >= len(self.views):
            return {
                "score": 0.0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "weak_reference": True,
            }

        ref = self.views[view_index]
        if ref.descriptors is None or len(ref.keypoints) < 12:
            return {
                "score": 0.0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "weak_reference": True,
            }

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = self.sift.detectAndCompute(gray, None)

        if desc2 is None or kp2 is None or len(kp2) < 8:
            return {
                "score": 0.0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "weak_reference": False,
            }

        try:
            matches = self.flann.knnMatch(ref.descriptors, desc2, k=2)
        except cv2.error:
            return {
                "score": 0.0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "weak_reference": False,
            }

        good = []
        for pair in matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.72 * n.distance:
                good.append(m)

        good_count = len(good)
        if good_count < 5:
            count_score = min(1.0, good_count / 18.0)
            return {
                "score": 0.30 * count_score,
                "good_matches": good_count,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "weak_reference": False,
            }

        src = np.float32(
            [ref.keypoints[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst = np.float32(
            [kp2[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        inliers = 0
        inlier_ratio = 0.0

        if good_count >= 8:
            H, mask = cv2.findHomography(
                src,
                dst,
                cv2.RANSAC,
                4.0,
            )
            if mask is not None:
                inliers = int(mask.ravel().sum())
                inlier_ratio = inliers / max(1, good_count)

        count_score = min(1.0, good_count / 28.0)
        inlier_score = min(1.0, inliers / 18.0)

        score = (
            0.35 * count_score
            + 0.25 * inlier_score
            + 0.40 * inlier_ratio
        )

        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "good_matches": good_count,
            "inliers": inliers,
            "inlier_ratio": float(inlier_ratio),
            "weak_reference": False,
        }

    def score_candidate(
        self,
        crop: np.ndarray,
        yoloe_component: float = 0.0,
        class_match: float = 0.5,
        embedding: Optional[torch.Tensor] = None,
    ) -> dict:
        dino_raw, view_idx = self.dino_similarity(
            crop=crop,
            embedding=embedding,
        )
        color = self.color_similarity(crop, view_idx)
        shape = self.shape_similarity(crop, view_idx)
        local = self.sift_geometry(crop, view_idx)

        # DINO cosine is kept as the main identity cue.
        #
        # DINO values for unrelated objects can still be positive, so the
        # final decision is NOT based on DINO alone.
        if local["weak_reference"]:
            final = (
                0.68 * dino_raw
                + 0.14 * color
                + 0.08 * shape
                + 0.07 * yoloe_component
                + 0.03 * class_match
            )
        else:
            final = (
                0.56 * dino_raw
                + 0.22 * local["score"]
                + 0.09 * color
                + 0.05 * shape
                + 0.05 * yoloe_component
                + 0.03 * class_match
            )

        return {
            "score": float(np.clip(final, 0.0, 1.0)),
            "dino": float(dino_raw),
            "color": float(color),
            "shape": float(shape),
            "sift": float(local["score"]),
            "sift_good": int(local["good_matches"]),
            "sift_inliers": int(local["inliers"]),
            "sift_inlier_ratio": float(local["inlier_ratio"]),
            "weak_reference": bool(local["weak_reference"]),
            "view_index": int(view_idx),
            "yoloe_component": float(yoloe_component),
            "class_match": float(class_match),
        }
