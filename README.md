# Clinical-Grade Diabetic Retinopathy Detection using Transfer Learning

An end-to-end medical computer vision pipeline built with TensorFlow/Keras to automate the screening and severity staging of Diabetic Retinopathy (DR) from digital fundus images. 

Designed specifically with clinical utility, resource efficiency, and high-recall metrics in mind to prevent avoidable blindness.

---

##  Project Architecture Overview

This repository is built using clean, modular software engineering principles instead of monolithic notebooks, ensuring production readiness.

```text
diabetic-retinopathy-detection/
├── .gitignore             # Prevents heavy data/weights from cluttering version control
├── requirements.txt       # Production library dependencies
├── src/                   # Python production modules
│   ├── preprocessing.py   # Domain-specific image normalization (Ben Graham's pipeline)
│   ├── model.py           # Deep learning architecture (EfficientNetB0 Transfer Learning)
│   ├── train.py           # Optimization configurations & resource-saving callbacks
│   └── predict.py         # Clinical inference pipeline mapping numbers to diagnoses
