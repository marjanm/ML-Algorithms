# ML Learning Project — Progress Tracker

**Legend:** ✅ Done (code + output exists) · ❌ Not done yet · ⏭️ Skipped (write-up only, in the [ML Reference doc][mlref])

> The **[ML Reference doc][mlref]** referenced throughout is `A review of Machine Learning models.md`, which lives in the sibling **Engineering-Handbook** repo.

---

## Part I — Foundations

### 1. Core Concepts

- ✅ Bias-Variance Tradeoff → `core_concepts/bias_variance/`
- ✅ Gradient Descent (visual) → `core_concepts/gradient_descent/`
- ✅ Cross-Validation → `core_concepts/cross_validation/`
- ✅ Regularization (L1/L2) → `core_concepts/regularization/`
- ✅ Feature Engineering → `core_concepts/feature_engineering/`
- ✅ Hyperparameter Tuning → `core_concepts/hyperparameter_tuning/`
- ⏭️ Loss Functions (CE, Focal, Contrastive) → [ML Reference doc][mlref]

### 2. From-Scratch Implementations

- ✅ Logistic Regression → `from_scratch/logistic_regression/`
- ✅ Mini-batch Gradient Descent → `from_scratch/mini_batch_gd/`
- ✅ Softmax → `from_scratch/softmax/`
- ✅ Decision Tree Split → `from_scratch/decision_tree_split/`
- ✅ kNN → `from_scratch/knn/`
- ✅ TF-IDF → `from_scratch/tfidf/`
- ✅ Evaluation Metrics (AUC, P/R) → `compare_models.py`
- ✅ Embedding Similarity Search → `deep_learning/nlp_concepts/embeddings/`

---

## Part II — Supervised Learning

### 3. Linear Models

- ✅ Logistic Regression → `supervised/logistic_regression/`
- ✅ Linear Regression → `supervised/linear_regression/`
- ✅ Naive Bayes → `supervised/naive_bayes/`

### 4. Tree-Based Models

- ✅ Decision Tree → `supervised/decision_tree/`
- ✅ Random Forest → `ensemble/random_forest/`
- ✅ AdaBoost → `ensemble/adaboost/`
- ✅ XGBoost → `ensemble/xgboost_model/`
- ✅ LightGBM → `ensemble/lightgbm_model/`
- ✅ CatBoost → `ensemble/catboost_model/`

### 5. Other Classifiers

- ✅ KNN → `supervised/knn/`
- ✅ SVM → `supervised/svm/`
- ✅ Model Comparison (RF vs XGB vs KNN) → `compare_models.py`

---

## Part III — Unsupervised Learning

### 6. Clustering

- ✅ K-Means → `unsupervised/kmeans/`
- ✅ DBSCAN → `unsupervised/dbscan/`
- ✅ Hierarchical Clustering → `unsupervised/hierarchical_clustering/`
- ✅ Gaussian Mixture Models → `unsupervised/gmm/`
- ✅ Expectation-Maximization → `unsupervised/expectation_maximization/`

### 7. Dimensionality Reduction

- ✅ PCA → `unsupervised/pca/`
- ✅ t-SNE / UMAP → `unsupervised/tsne_umap/`

### 8. Anomaly Detection

- ✅ Isolation Forest / One-Class SVM / LOF → `anomaly_detection/`

---

## Part IV — Deep Learning

### 9. Neural Network Architectures

- ✅ MLP → `deep_learning/mlp/`
- ✅ CNN (MNIST) → `deep_learning/cnn/`
- ✅ RNN / LSTM (sine wave) → `deep_learning/rnn_lstm/`

### 10. Generative Models

- ✅ GAN (MNIST) → `deep_learning/gan/`
- ✅ Autoencoder / VAE (MNIST) → `deep_learning/autoencoder_vae/`
- ✅ Diffusion Model (MNIST) → `deep_learning/diffusion/`

### 11. Transformers & Language Models

- ✅ Transformer (from scratch) → `deep_learning/transformer/`
- ✅ GPT-2 (pre-trained) → `deep_learning/transformer/`
- ✅ BERT (pre-trained) → `deep_learning/bert/`

---

## Part V — NLP & LLM Concepts

### 12. Text Representations

- ✅ Tokenization → `deep_learning/nlp_concepts/tokenization/`
- ✅ Embeddings → `deep_learning/nlp_concepts/embeddings/`
- ✅ Attention Mechanism → `deep_learning/transformer/`

### 13. Transfer Learning & Adaptation

- ✅ Fine-tuning vs Transfer Learning → `deep_learning/nlp_concepts/fine_tuning/`
- ✅ LoRA / QLoRA → `deep_learning/nlp_concepts/lora/`
- ✅ RLHF → [ML Reference doc][mlref] (conceptual only)

### 14. Retrieval & Generation

- ✅ RAG → `deep_learning/nlp_concepts/rag/`

---

## Part VI — Applied ML & Practical Skills

### 15. Evaluation & Experimentation

- ✅ Evaluation Metrics Deep Dive → `evaluation_metrics/`
- ✅ A/B Testing / Statistical Tests → `ab_testing/`
- ✅ Explainability (SHAP / LIME) → `explainability/`

### 16. Data & Preprocessing

- ✅ Data Preprocessing Pipeline → `data_preprocessing/`
- ✅ Data Augmentation → `deep_learning/data_augmentation/`
- ⏭️ Data Leakage → [ML Reference doc][mlref]
- ⏭️ Feature Crossing → [ML Reference doc][mlref]

### 17. Specialized Applications

- ✅ Recommendation Systems → `recommendation_systems/`
- ⏭️ Content-Based vs Collaborative Filtering → [ML Reference doc][mlref]
- ⏭️ Hybrid Systems → [ML Reference doc][mlref]
- ⏭️ Matrix Factorisation (SVD) → [ML Reference doc][mlref]
- ⏭️ Two-Tower Model → [ML Reference doc][mlref]
- ⏭️ WALS vs SGD → [ML Reference doc][mlref]
- ⏭️ Exploration vs Exploitation → [ML Reference doc][mlref]
- ⏭️ Ranking Metrics (MRR, MAP, NDCG, Recall@K) → [ML Reference doc][mlref]
- ✅ Time Series (classical: ARIMA) → `time_series_classical/`
- ✅ Reinforcement Learning (Q-Learning / DQN) → `reinforcement_learning/`
- ⏭️ Multi-label Classification → [ML Reference doc][mlref]
- ⏭️ Multi-task Learning → [ML Reference doc][mlref]

---

## Part VII — ML Theory Deep Dives

### 18. Learning with Limited Labels

- ✅ Semi-supervised / Self-supervised → `semi_supervised/`
- ✅ Active Learning → `active_learning/`
- ⏭️ Human-in-the-Loop → [ML Reference doc][mlref]

### 19. Model Efficiency

- ✅ Model Distillation → `model_distillation/`
- ✅ Continual / Lifelong Learning → `continual_learning/`

### 20. Causality & Fairness

- ✅ Causal Inference → `causal_inference/`
- ✅ Ethics / Fairness / Bias → `fairness_bias/`
- ⏭️ Propensity Scoring / Bias Correction → [ML Reference doc][mlref]

---

## Part VIII — MLOps & Production

### 21. Deployment & Infrastructure

- ✅ Model Serving (FastAPI) → `model_serving/`
- ✅ Containerization (Docker) → `model_serving/Dockerfile`
- ✅ CI/CD for ML Pipelines → `.github/workflows/ml_pipeline.yml`
- ✅ Experiment Tracking (MLflow) → `experiment_tracking/`
- ✅ Model Registry Lifecycle → `experiment_tracking/mlflow_registry_monitoring.py`
- ✅ Reproducibility Guarantees → `reproducibility/`
- ⏭️ Cloud ML Services → [ML Reference doc][mlref] (SageMaker / Vertex AI / Azure ML comparison)

### 22. Monitoring & Reliability

- ✅ Feature Drift Detection → `drift_detection/` (PSI, KS, KL divergence)
- ✅ Model Monitoring Dashboard → `experiment_tracking/mlflow_registry_monitoring.py` (MLflow UI)
- ✅ Retraining Trigger Strategy → `drift_detection/` + `experiment_tracking/` (PSI threshold → retrain)
- ✅ Monitoring Silent Degradation → `drift_detection/` (accuracy vs drift over 12 weeks)

### 23. System Design

- ✅ Offline vs Online Gap Case Study → `offline_online_gap/`
- ✅ Cold Start Strategy → `recommendation_systems/cold_start_demo.py`
- ✅ Latency-Constrained Inference → `latency_benchmark/`
- ✅ Extreme Imbalance Ranking System → `extreme_imbalance/`
- ✅ Calibration (Platt Scaling + Isotonic) → `calibration/`
- ✅ Cost-Sensitive Learning → `cost_sensitive/`
- ⏭️ Feature Store Design → [ML Reference doc][mlref]
- ⏭️ Full ML System Architecture Diagram → [ML Reference doc][mlref]
- ⏭️ Batch vs Real-time Inference → [ML Reference doc][mlref]
- ⏭️ ML Failure Modes Document → [ML Reference doc][mlref]
- ⏭️ Data-Centric ML Workflow → [ML Reference doc][mlref]
- ⏭️ A/B Testing Design + Pitfalls → [ML Reference doc][mlref]

---

## Part IX — Frontiers

### 24. LLM-Era / Modern AI

- ✅ AI Agents / Tool Use → `ai_agents/`
- ✅ Multi-modal Models (CLIP) → `multimodal_clip/`
- ✅ LLM Evaluation (BLEU, ROUGE) → `llm_evaluation/`
- ✅ Vector Databases (FAISS, ANN) → `vector_databases/`
- ⏭️ Guardrails / LLM Safety → [ML Reference doc][mlref]

### 25. Advanced Theory

- ✅ Probabilistic Modeling Intuition → `probabilistic_modeling/`
- ✅ Uncertainty Estimation (MC Dropout, Deep Ensembles) → `uncertainty_estimation/`
- ✅ Robustness & Adversarial Thinking → `adversarial_robustness/`
- ⏭️ Data Validation Frameworks → [ML Reference doc][mlref]

### 26. Scalability

- ⏭️ Distributed Training Basics → [ML Reference doc][mlref]
- ❌ Memory Optimization for Large Models
- ✅ Approximate Nearest Neighbor Search → `vector_databases/` (merged with Vector DBs)
- ⏭️ ANN Deep Dive (LSH, PQ, ScaNN, curse of dimensionality) → [ML Reference doc][mlref]
- ⏭️ Caching Strategies for Inference → [ML Reference doc][mlref]
- ⏭️ Throughput vs Latency Tradeoffs → [ML Reference doc][mlref]
- ⏭️ Federated Learning → [ML Reference doc][mlref]

### 27. Debugging & Diagnostics

- ✅ Structured Error Slicing → `error_slicing/`
- ⏭️ Counterfactual Reasoning → [ML Reference doc][mlref]
- ⏭️ Root Cause Analysis for ML Failures → [ML Reference doc][mlref]

### 28. Generalization & Robustness

- ⏭️ OOD Generalization (Out-of-Distribution) → [ML Reference doc][mlref]

---

## Skipped (doc-only)

- ⏭️ Prompt Engineering
- ⏭️ Object Detection (YOLO)
- ⏭️ Graph Neural Networks

---

## Reference Notes

See the **[ML Reference doc][mlref]** for detailed write-ups on all topics, plus:

- Loss Functions (Cross-Entropy, Focal, Class-Balanced, Contrastive)
- Convex Optimization / KKT
- Functional Gradient Descent / Why Boosting Works
- Spark Fundamentals for ML Engineers
- Non-Technical Skills (product goals, KPIs, design docs, stakeholder communication)
- Guardrails / LLM Safety
- Data Validation Frameworks
- Distributed Training Basics
- Caching Strategies for Inference
- Throughput vs Latency Tradeoffs
- ANN Deep Dive (index types, vector quantization, ScaNN, curse of dimensionality)
- Counterfactual Reasoning
- Root Cause Analysis for ML Failures
- Feature Store Design
- ML System Architecture
- ML Failure Modes
- Batch vs Real-time Inference
- Data-Centric ML
- A/B Testing Design
- Data Leakage
- Feature Crossing
- Multi-label Classification
- Multi-task Learning
- Human-in-the-Loop
- Propensity Scoring / Bias Correction
- Federated Learning
- OOD Generalization
- Recommendation Systems Deep Dive (two-tower, WALS vs SGD, hybrid, ranking metrics)

---

## 📌 Pinned — To Be Coded Later

- ❌ Ranking Metrics Demo (NDCG, MAP, MRR, Recall@K) → `recommendation_systems/`
- ❌ Bandit Explore-Exploit Demo (ε-greedy, UCB, Thompson Sampling) → `recommendation_systems/`

---

## 📌 Gaps From Scikit-Learn Guide — To Be Covered

- ❌ Ridge / Lasso / Elastic-Net Comparison → `supervised/regularized_regression/`
  *Common interview topic. Show L1 vs L2 vs combined, coefficient shrinkage, feature selection behavior.*
- ❌ Gaussian Processes (GPR / GPC) → `supervised/gaussian_processes/`
  *Uncertainty-aware model. Returns confidence intervals natively. Good for small data + Bayesian optimization.*
- ❌ Feature Selection (RFE, SelectFromModel, Sequential) → `core_concepts/feature_selection/`
  *Practical skill. Which features to keep? Recursive elimination, model-based, forward/backward search.*
- ❌ Sklearn Pipelines & ColumnTransformer → `core_concepts/pipelines/`
  *Critical production skill. Chain preprocessing + model. Prevents data leakage. ColumnTransformer for mixed types.*
- ❌ Validation Curves & Learning Curves → `core_concepts/learning_curves/`
  *Diagnostic tool. Is the model underfitting or overfitting? How much more data would help?*
- ❌ Missing Value Imputation Strategies → `data_preprocessing/imputation/`
  *Every real dataset has missing values. KNN imputation, iterative (MICE), simple strategies, impact comparison.*
- ❌ LDA / QDA (Discriminant Analysis) → `supervised/discriminant_analysis/`
  *Classic Bayesian classifiers. LDA = linear boundary, QDA = quadratic. Good when class distributions are Gaussian.*
- ❌ Multiclass Strategies (OvR, OvO) → `core_concepts/multiclass/`
  *How binary classifiers handle 3+ classes. One-vs-Rest vs One-vs-One. When each is better.*
- ❌ HDBSCAN / OPTICS → `unsupervised/density_clustering/`
  *Extensions of DBSCAN. HDBSCAN = no epsilon needed, handles varying density. OPTICS = reachability plots.*
- ❌ ICA / NMF → `unsupervised/matrix_factorization/`
  *Beyond PCA. ICA = find independent signals (blind source separation). NMF = non-negative parts-based decomposition.*

[mlref]: https://github.com/marjanm/Engineering-Handbook/blob/main/A%20review%20of%20Machine%20Learning%20models.md
