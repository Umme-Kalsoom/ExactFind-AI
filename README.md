# ExactFind AI — High Accuracy Version

This version keeps the same working user experience:

- Capture target OR upload target image(s)
- Live camera stays visible
- Normal detected objects = GREEN boxes
- Confirmed exact physical target = RED box
- No whole-camera red box

The main change is the verification engine.

## Accuracy upgrades

1. **DINOv2 ViT-B/14 with registers**
   - Main exact-instance appearance feature.
2. **SIFT + RANSAC**
   - Checks small local details and geometric consistency.
3. **Multi-reference support**
   - Capture or upload 2–4 views of the SAME physical object.
4. **YOLOE visual prompt**
   - Used as an additional target cue, not the only gate.
5. **Normal YOLO scene candidates**
   - Every visible object box is independently verified.
6. **Top-1 / top-2 ambiguity rejection**
   - If two objects look almost equally similar, neither becomes red.
7. **3-frame temporal confirmation**
   - A one-frame false match does not become the target.
8. **Whole-frame target rejection**
   - Broad YOLOE boxes cannot turn the whole camera red.

## Important

The percentage next to `TARGET` is a live match score, not dataset accuracy.

Real accuracy / precision / recall / F1 require labeled test frames.

## Install on Kali / Ubuntu

```bash
sudo apt update
sudo apt install -y python3-tk

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

The first run can download:
- YOLO model weights
- YOLOE model weights
- official DINOv2 repository/weights through PyTorch Hub

## Best workflow for accuracy

### Capture
1. Put the target in the yellow box.
2. Click `1. Capture Target`.
3. Rotate it slightly.
4. Click `Add Target View`.
5. Repeat for 2–4 views.
6. Click `2. Start Search`.

### Upload
You can select 1–5 images at once. They must all show the SAME physical target.

## Accuracy-first settings

Default:

```bash
python app.py
```

Fewer false positives:

```bash
python app.py \
  --verify-threshold 0.76 \
  --dino-min 0.74 \
  --ambiguity-margin 0.07 \
  --confirm-frames 4
```

If the real target is frequently missed:

```bash
python app.py \
  --verify-threshold 0.68 \
  --dino-min 0.66 \
  --ambiguity-margin 0.04 \
  --confirm-frames 2
```

## CPU mode

The default DINOv2-B model is accuracy-oriented and can be slower on CPU.
For faster CPU inference:

```bash
python app.py --dino-model dinov2_vits14_reg --scene-model yolo11n.pt
```

Keep the default ViT-B model when your main priority is accuracy
