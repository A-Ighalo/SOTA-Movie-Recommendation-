# Fair and Equitable AI in Recommendation Systems

A comprehensive implementation and evaluation of fair and equitable recommendation systems using multiple machine learning approaches. This project explores content-based filtering, collaborative filtering, popularity-based recommendations, and hybrid models while prioritizing fairness and avoiding algorithmic bias.

##  Project Overview

Recommendation systems play a central role in shaping user experiences across digital platforms. This project focuses on building recommendation systems that operate in a **fair and equitable manner**, avoiding unjust or systematic bias while promoting inclusivity through algorithmic design.

### 🔑 Key Features

- **Multiple Recommendation Approaches**: Popularity-based, Content-based, Collaborative Filtering (SVD), and Hybrid models
- **Comprehensive Evaluation**: Precision, Recall, F1-score, NDCG, MAP, MRR, and Diversity metrics
- **MLflow Integration**: Complete experiment tracking and model management
- **Fairness Focus**: Designed to minimize demographic bias and promote equitable recommendations
- **Real-world Dataset**: Amazon sales dataset with 1,465 products and comprehensive user reviews

## ℹ️ Dataset

The project uses the **Amazon Sales Dataset** containing:
- **1,465 products** across 211 unique categories
- **1,194 unique users** with comprehensive review data
- Product features: pricing, ratings, categories, descriptions
- User interaction data: ratings, reviews, preferences


## 📁 Project Structure once run on google colab

```plaintext
├── Drive/               # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── mlruns/              # MLflow experiment logs
├── README.md            # Project documentation
└── requirements.txt     # Dependencies

```
## 🧑‍💻 Tech Stack

- **Python 3.7+**
- **Machine Learning**: scikit-learn, scikit-surprise
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Experiment Tracking**: MLflow
- **Text Processing**: TF-IDF vectorization
- **Environment**: Google Colab

##  Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Dependencies
```bash
# Core ML libraries
pip install pandas numpy scikit-learn scikit-surprise

# Visualization
pip install matplotlib seaborn

# Experiment tracking
pip install mlflow

# Text processing
pip install nltk

# Additional utilities
pip install tabulate kagglehub pyngrok
```

### Google Colab Setup
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install mlflow scikit-surprise tabulate pyngrok
```

## 📈 Recommendation Systems Implemented

### 1. Popularity-Based Recommendation
- **Approach**: Recommends most popular items based on rating count and user engagement
- **Metrics**: Hit Rate: 0.90, Precision: 1.00, Recall: 0.94
- **Use Case**: Cold start problems, trending items

### 2. Content-Based Filtering
- **Approach**: Uses TF-IDF vectorization of product categories and descriptions
- **Similarity**: Cosine similarity between item features
- **Metrics**: Precision: 0.30, Recall: 1.00, F1: 0.46, NDCG: 0.88
- **Strength**: No cold start problem for items

### 3. Collaborative Filtering (SVD)
- **Approach**: Matrix factorization using Singular Value Decomposition
- **Metrics**: RMSE: 0.26, Precision: 0.98, Recall: 0.98, F1: 0.98
- **Strength**: Captures complex user-item interactions

### 4. Hybrid Approaches
- **Simple Hybrid**: Union of content-based and collaborative recommendations
- **Weighted Hybrid**: Configurable α parameter to balance approaches
- **Advantage**: Combines strengths of multiple methods

## 🔬 Evaluation Metrics

### Accuracy Metrics
- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **F1-Score**: Harmonic mean of precision and recall
- **RMSE**: Root Mean Squared Error for rating prediction

### Ranking Metrics
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank

### Diversity & Fairness
- **Intra-list Diversity**: Measures recommendation variety
- **Coverage**: Ensures diverse item representation
- **Fairness Evaluation**: Bias detection across demographics

## 🎪 Usage Examples

### Quick Start
```python
# Load and preprocess data
df = pd.read_csv("amazon.csv")
ratings_df = preprocess_data(df)

# Content-based recommendations
recommendations = get_recommendations(product_index=1, n=10)

# Collaborative filtering
algo = SVD()
algo.fit(trainset)
user_recommendations = get_top_n_recommendations(user_id, n=10)

# Hybrid approach
hybrid_recs = hybrid_recommendations(user_id, product_index, algo, alpha=0.6)
```

### MLflow Experiment Tracking
```python
# Start MLflow experiment
mlflow.set_experiment("Recommender_System")

with mlflow.start_run(run_name="Content_Based_RecSys"):
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_artifact("recommendations.json")
```

## 🕵️ Results Summary

| Model | Precision | Recall | F1-Score | RMSE | Diversity |
|-------|-----------|--------|----------|------|-----------|
| Popularity-Based | 1.00 | 0.94 | 0.97 | - | Medium |
| Content-Based | 0.30 | 1.00 | 0.46 | - | 0.27 |
| Collaborative (SVD) | 0.98 | 0.98 | 0.98 | 0.26 | High |
| Weighted Hybrid | - | - | - | - | High |

## 🔍 Key Findings

1. **Collaborative Filtering** achieved the highest overall performance with 98% precision and recall
2. **Content-Based** systems excel at diversity but may sacrifice precision
3. **Hybrid approaches** successfully balance accuracy and diversity
4. **Popularity-based** systems are effective for trending recommendations but may lack personalization

## 😇 Fairness Considerations

This project specifically addresses:
- **Demographic Bias**: Avoiding recommendations based on protected attributes
- **Popularity Bias**: Balancing popular vs. niche item recommendations
- **Filter Bubbles**: Promoting diverse content discovery
- **Equal Opportunity**: Ensuring fair access to recommendations across user groups


### Areas for Contribution
- Additional fairness metrics implementation like fairness 360
- New recommendation algorithms
- Performance optimizations
- Documentation improvements
- Dataset expansion
- Productiion grade python files

## 📚 References

- Ekstrand, M. NSF CAREER award on recommenders, humans, and data
- Gunawardana, A., & Shani, G. (2009). A survey of accuracy evaluation metrics of recommendation tasks
- Wang, Y., et al. (2023). A survey on the fairness of recommender systems
- Abdollahpouri, H., & Burke, R. (2019). Multi-stakeholder recommendation and its connection to multi-sided fairness

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<img width="1271" height="426" alt="image" src="https://github.com/user-attachments/assets/320600b4-3672-4445-952e-8a8f99e0122e" />
