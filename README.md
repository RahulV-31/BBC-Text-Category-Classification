### BBC Text Category Classification  
---

**Project Overview**  
This project focuses on classifying BBC news articles into five predefined categories using text preprocessing, TF-IDF vectorization, and machine learning models. The project demonstrates effective text classification and offers practical applications in content organization and analysis.  
---

**Dataset**  
- **Shape**:  
  - **Number of Documents**: 2,225  
  - **Number of Categories**: 5  
- **Categories in the Dataset**:  
  - ['business', 'entertainment', 'politics', 'sport', 'tech']  
---

**Objective**  
The project's primary goals are to:  
1. Preprocess and clean the dataset for effective analysis.  
2. Transform textual data into numerical representations using TF-IDF.  
3. Build and evaluate machine learning models for category prediction.  
4. Develop a real-time classification system using Flask for user input text.  
---

**Project Workflow**  

1. **Data Preprocessing**:  
   - Cleaned text by removing punctuation, converting to lowercase, and removing stop words.  
   - Applied stemming and lemmatization for normalization.  

2. **Feature Extraction**:  
   - Utilized TF-IDF vectorization to represent textual data numerically.  

3. **Modeling**:  
   - Implemented two machine learning models:  
     - **Naive Bayes Classifier**  
       - Accuracy: **97.98%**  
       - Classification Report:  
         - High precision and recall for all categories, achieving F1-scores of 0.96-1.00.  
     - **Logistic Regression**  
       - Accuracy: **97.98%**  
       - Classification Report:  
         - Strong performance across all categories, with F1-scores ranging from 0.97-1.00.  

4. **Application Development**:  
   - Created a Flask application for real-time classification.  
   - Users can input text, and the app predicts its category based on trained models.  
---

**Results**  
- Both models performed exceptionally, with nearly identical accuracy and classification metrics.  
- The classification reports for Naive Bayes and Logistic Regression showcase strong category-wise performance.
---

**Tools and Technologies**  
- **Programming Language**: Python  
- **Libraries**:  
  - Preprocessing: NLTK, scikit-learn  
  - Model Development: scikit-learn  
  - Web Framework: Flask  
  - Frontend: HTML, CSS
---  

**Key Metrics**  
- **Naive Bayes Accuracy**: **97.98%**  
- **Logistic Regression Accuracy**: **97.98%**  
- Both models demonstrate robust performance with minimal differences in evaluation metrics.  

This project highlights the potential of machine learning and NLP for efficient and accurate text categorization.
--
