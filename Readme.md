# Face Detection, Recognition & Clustering

> An end-to-end Computer Vision pipeline benchmarking classical and deep methods for face detection, recognition, and identity clustering — finishing at **98.6%** detection and **95%** recognition accuracy on the project test set.

Built as the major project for the Computer Vision course at **IIT Jodhpur (Mar–Apr 2024)**. The repo bundles three tightly related sub-projects, each comparing a different family of approaches end-to-end so the trade-offs are visible side-by-side.

---

## Headline results

| Task | Best Method | Accuracy |
|---|---|---|
| Face Detection | MTCNN | **98.60%** |
| Face Recognition | CNN embeddings (FaceNet-style) | **95.00%** |
| Face Clustering | DBSCAN over deep embeddings | qualitative — clean per-identity clusters |

Methods benchmarked: **Viola-Jones · HOG · SSD · YOLO · Faster R-CNN · MTCNN · FaceNet · Autoencoder · PCA · LDA**

---

## Repository layout

```
CV-MajorProject/
├── face-clustering-app/         # 1. Production-style Flask web app
├── faceDetection&Clustering/    # 2. Deep-learning detection + clustering
└── FaceDetectionPCA+LDA/        # 3. Classical baseline (PCA + LDA)
```

### 1 · `face-clustering-app/` — Web app

Flask app that accepts a folder of photos, encodes every face with `dlib`'s deep-learning face encoder, and clusters identities with DBSCAN. Each cluster shows up as a gallery of every photo the person appears in.

```bash
cd face-clustering-app
pip install -r requirements.txt
python encode_faces.py --dataset dataset1   # generate encodings.pickle
python cluster_faces1.py                    # run DBSCAN clustering
python app.py                               # launch Flask UI on :5000
```

### 2 · `faceDetection&Clustering/` — Deep-learning pipeline

Notebooks comparing detection backbones (**SSD, YOLO, Faster R-CNN, MTCNN**) and clustering strategies (**FaceNet embeddings + DBSCAN** vs. **Autoencoder bottleneck + KMeans**). Best detector: MTCNN at 98.6% on the held-out set.

### 3 · `FaceDetectionPCA+LDA/` — Classical baseline

Notebook implementing the eigenfaces / fisherfaces classical pipeline so you can see exactly how much accuracy a deep model buys you over PCA + LDA on the same dataset.

---

## Stack

| Category | Tools |
|---|---|
| Detection | OpenCV (Haar/Viola-Jones, HOG), MTCNN, YOLO, SSD, Faster R-CNN |
| Recognition / Embedding | dlib, FaceNet, Autoencoders, PCA, LDA |
| Clustering | DBSCAN, KMeans |
| Web app | Flask, Pillow |
| Notebooks | NumPy, scikit-learn, OpenCV, PyTorch |

---

## Quickstart (entire repo)

```bash
git clone https://github.com/Arun-Raghav-S/CV-MajorProject.git
cd CV-MajorProject

# Pick the sub-project you want, then follow its README
cd face-clustering-app && cat README.md
```

Each sub-folder ships with its own README explaining datasets, training, and inference.

---

## What I learned building it

- **MTCNN beats classical detectors comfortably**, but classical methods (Viola-Jones, HOG) still earn their keep on resource-constrained devices because of their 10–100× lower inference cost.
- **Embedding-based clustering (DBSCAN over FaceNet)** generalises far better than autoencoder bottlenecks for unseen identities — autoencoders overfit to training-set faces.
- **PCA + LDA** is a surprisingly strong baseline at <1% the compute of any deep model — useful for sanity-checking the value of a heavier stack before committing to it.

---

— [Arun Raghav S](https://github.com/Arun-Raghav-S) · [arunraghavdev.com](https://arunraghavdev.com/)
