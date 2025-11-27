#  Visual Embedding Analysis for X-rays of Cats & Dogs Dataset

https://github.com/user-attachments/assets/5a4c06f8-8263-44c4-abf0-a6a73c7edb77

This project explores how modern foundation vision models represent medical X-ray images.  
We extract embeddings using **DINOv2**, combine them with **SAM2 segmentation**, and compare how well the representations cluster cats vs. dogs using **UMAP**, **t-SNE**, and quantitative metrics.

This notebook demonstrates:
- Clean preprocessing for medical X-rays (CLAHE, resizing, RGB conversion)  
- High-quality embeddings using **DINOv2 (ViT-S/14)**  
- Correct SAM2 workflow (segmentation → masked image → DINOv2 embedding)  
- Dimensionality reduction (UMAP + t-SNE)  
- Clustering quality evaluation (Silhouette Score, Davies–Bouldin Index)  
- Static and interactive visualizations  
- Full reproducibility with structured outputs


## Project Structure

```
AI4See-Assignment/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
│
├── processed_images/
├── outputs/
│ ├── embeddings/
│ ├── plots/
│ └── interactive/
│
└── test-dataset/
├── cat1/
└── dog1/
```


## Methodology

### **1. Preprocessing**
- Load X-ray images in grayscale  
- Apply **CLAHE** (improves X-ray contrast)  
- Resize → Convert to RGB  
- Normalize to ImageNet statistics (for DINOv2)

### **2. Embedding Extraction**
#### **A. DINOv2 (Full Image)**
- Use `dinov2_vits14` from Facebook Research
- Extract 384-dimensional embeddings  
- Very fast on GPU

#### **B. SAM2 + DINOv2 (Segmented Version)**
- Use SAM2 to segment the primary anatomical region  
- Apply DINOv2 only on masked region  
- Produces embeddings with reduced background noise

### **3. Dimensionality Reduction**
- **UMAP** (cosine metric)  
- **t-SNE** (perplexity = 30, cosine metric)  
- Produces 2-D visual clusters

### **4. Clustering Metrics**
We compute:

| Method | Silhouette ↑ | Davies-Bouldin ↓ |
|--------|--------------|------------------|
| DINOv2 + UMAP | 0.0414 | 7.6403 |
| DINOv2 + t-SNE | 0.0693 | 5.0226 |
| SAM→DINOv2 + UMAP | 0.0058 | 17.3990 |
| SAM→DINOv2 + t-SNE | 0.0109 | 13.5544 |


## Key Insights

- **DINOv2 full-image embeddings outperform segmented SAM2 versions**  
  → SAM segmentation over-cropped anatomical details, weakening class separation.

- **t-SNE consistently produced slightly better separation than UMAP.**

- **Dataset is extremely challenging**  
  → Anatomical X-rays of cats and dogs look extremely similar.  
  → Even state-of-the-art models show low inter-class variance.

- **Pipeline is fully reproducible and works on CPU/GPU**, with GPU providing ~10× speedup.


## How to Run
1. Clone the repository  
2. Install dependencies:
```
pip install -r requirements.txt
```


3. Open the notebook:
```
jupyter notebook notebook.ipynb
```

4. Run all cells — outputs appear under `outputs/`.


## Demo
A short demonstration video is included showing:
- Notebook execution  
- Intermediate visual outputs  
- Final cluster comparison  

## Links
All the deliverables and sharable links are included:
- Notebook Link: https://colab.research.google.com/drive/1QxrMt-DPyDdHTktLDFrrVtK4USIKa3B4?usp=sharing
- Drive Link: https://drive.google.com/drive/folders/1LHgMlgd7VXSGdekH-5DNUzoaPyuYTi_j?usp=sharing  


## Author
**Mohammed Tawfiq**  
AI/ML Engineer — Computer Vision & GenAI  
