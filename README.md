# 🧠 Classifying TikTok Videos: Predicting Claims vs Opinions Using Machine Learning  

## 📘 Project Overview  
This project applies **Natural Language Processing (NLP)** and **Machine Learning** to classify TikTok videos as either a **claim** or an **opinion** based on their captions and metadata. The goal was to help TikTok’s **Trust and Safety** team automatically flag claim-based videos for moderation review.  

Using a dataset of over **10,000 TikTok video records**, several models—including **Random Forest** and **XGBoost**—were trained, tuned, and evaluated. The final model achieved **perfect recall (1.0)** and **precision (1.0)** on the validation set, indicating excellent classification performance.

---

## 💼 Business Understanding  
TikTok moderators face challenges identifying misinformation among millions of uploaded videos daily. Manually reviewing all videos is time-consuming and inconsistent.  

The goal of this project was to create an **automated classification model** to:  
- Identify factual **claims** that require human review.  
- Differentiate **opinions** to reduce false flags.  
- Improve efficiency and consistency of content moderation.  

This work supports **TikTok’s mission** to ensure the accuracy and reliability of user-generated content while maintaining user engagement and safety.

---

## 📊 Data Understanding  
- **Dataset Source:** Google Advanced Data Analytics Capstone Challenge (TikTok Dataset)  
- **Data Size:** 10,409 video records  
- **Key Features:**
  - `video_transcription_text` – Text captions from TikTok videos  
  - `verified_status` – Whether the creator is verified  
  - `video_duration`, `engagement_metrics` – Quantitative video features  
  - `claim_status` – Target label (claim or opinion)  
- **Data Cleaning & Feature Engineering:**
  - Removed duplicates and missing values  
  - Engineered `text_length` feature from transcription text  
  - Encoded categorical variables  
  - Balanced classes using `class_weight='balanced'`  

---

## 🤖 Modeling and Evaluation  

### Models Tested  
1. Logistic Regression  
2. Random Forest Classifier  
3. XGBoost Classifier  

### Model Tuning  
Hyperparameter tuning was performed using **GridSearchCV** with 5-fold cross-validation.  
Evaluation metrics focused on **recall** and **precision** to ensure balanced classification.

**Random Forest Best Parameters:**  
```python
{
  'n_estimators': 200,
  'max_depth': 10,
  'min_samples_split': 5,
  'min_samples_leaf': 2
}
