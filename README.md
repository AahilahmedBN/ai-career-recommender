# 🚀 Hybrid AI-Based Career Recommendation System

A multi-layered intelligent system that recommends careers based on student skill profiles using a combination of:

- K-Means Clustering (pattern discovery)
- Collaborative Filtering (similarity-based recommendation)
- Neural Networks (MLP for prediction)
- Skill-Fit Validation (real-world feasibility check)

---

## 📌 Problem Statement

Students often struggle to choose the right career path due to:
- Lack of personalized guidance
- Generic recommendation systems
- No validation of actual skill readiness

This project solves that by building a **hybrid AI system** that is both **data-driven and realistic**.

---

## 🧠 System Architecture

The system is designed as a **multi-stage decision pipeline**:

1. **Input Layer**
   - User enters skill scores (0–10):
     - Math
     - Creativity
     - Communication
     - Technical
     - Leadership
   - Selects an interest domain

2. **Clustering (K-Means)**
   - Groups users into archetypes based on skill patterns
   - Helps understand *“what type of student you are”*

3. **Collaborative Filtering**
   - Finds similar students using cosine similarity
   - Recommends careers based on peer patterns
   - Domain-restricted to avoid irrelevant suggestions

4. **Neural Network (MLP)**
   - Domain-specific models (not one global model)
   - Predicts probability of success in each career
   - Architecture:
     ```
     Input (5 features) → Dense(64) → Dense(32) → Output
     ```

5. **Skill-Fit Validation**
   - Checks if user meets minimum skill thresholds
   - Flags gaps to ensure realistic recommendations

---

## 📊 Dataset

- **Type:** Synthetic (generated programmatically)
- **Size:** 700+ student records
- **Careers:** 14
- **Domains:** 7 (AI, Tech, Design, Business, Science, Healthcare, Media)

### Why Synthetic Data?
Real labeled career datasets are not publicly available.  
Synthetic data allows:
- Controlled experimentation
- Clear separation between career profiles
- Model validation

---

## ⚙️ Tech Stack

- **Language:** Python
- **Libraries:**
  - `numpy`, `pandas`
  - `scikit-learn`
  - `matplotlib`

---

## 📈 Machine Learning Components

### 1. K-Means Clustering
- Unsupervised learning
- Elbow Method used to select optimal K = 7
- Discovers hidden student archetypes

---

### 2. Collaborative Filtering
- Uses **cosine similarity**
- Finds top 50 similar profiles
- Recommends most frequent careers among them

---

### 3. Neural Network (MLP)
- Separate model per domain
- Prevents cross-domain misclassification
- Uses:
  - ReLU activation
  - Adam optimizer
  - Early stopping

---

### 4. Skill-Fit Gate
- Validates recommendations against minimum skill thresholds
- Adds practical realism to predictions

---

## 📊 Output

For each user, the system provides:

- 🎯 **Student Archetype** (Cluster)
- 💼 **Top Career Recommendations**
- 📉 **Probability Scores (MLP)**
- ⚠️ **Skill Gap Analysis**

---

## 📉 Visualization Dashboard

The system generates a dashboard with:

- Elbow Curve (optimal K selection)
- PCA Cluster Visualization
- Career Distribution per Cluster
- MLP Probability Chart

---

## 🧪 Example Input

| Skill         | Score |
|--------------|------|
| Math         | 9    |
| Creativity   | 4    |
| Communication| 4    |
| Technical    | 9    |
| Leadership   | 3    |

**Expected Output:**
- AI/ML Engineer (High Confidence)
- Data Scientist

---
