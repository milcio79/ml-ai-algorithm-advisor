from __future__ import annotations

from dataclasses import dataclass

from models import AlgorithmRecommendation, ProjectProfile


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    category: str
    triggers: tuple[str, ...]
    why: str
    best_when: str
    cautions: str
    required_data: str
    metrics: str
    notes: str
    base_score: int = 18


def _spec(
    name: str,
    category: str,
    triggers: list[str],
    why: str,
    best_when: str,
    cautions: str,
    required_data: str,
    metrics: str,
    notes: str,
    base_score: int = 18,
) -> AlgorithmSpec:
    return AlgorithmSpec(
        name=name,
        category=category,
        triggers=tuple(triggers),
        why=why,
        best_when=best_when,
        cautions=cautions,
        required_data=required_data,
        metrics=metrics,
        notes=notes,
        base_score=base_score,
    )


CLASSIFICATION = [
    "Predict a known category or class",
    "Target variable is categorical",
    "Target variable is binary",
    "Target has multiple classes",
    "Reliable labeled data is available",
    "Historical outcomes are available",
    "Human review outcomes are available",
]
REGRESSION = [
    "Predict a numeric value",
    "Target variable is continuous",
    "Reliable labeled data is available",
    "Historical outcomes are available",
    "Human review outcomes are available",
]
EXPLAINABLE = [
    "High explainability required",
    "Auditability required",
    "Traceability required",
    "Need business-friendly explanation",
]
PRODUCTION = [
    "Production deployment required",
    "MLOps required",
    "Model monitoring required",
    "Data drift is expected",
    "Need versioned decisions",
]


CATALOG: list[AlgorithmSpec] = [
    _spec("Logistic Regression", "General ML / Structured Data", CLASSIFICATION + EXPLAINABLE + ["Structured tabular data"], "Interpretable supervised classifier for binary or multiclass outcomes.", "Labels are available and decision drivers must be explainable.", "May underfit nonlinear relationships without feature engineering.", "Labeled tabular data with categorical or binary target.", "Accuracy, F1, ROC-AUC, PR-AUC, precision, recall.", "Use regularization, calibrated probabilities, threshold tuning, and SHAP/coefficient review."),
    _spec("Linear Regression", "General ML / Structured Data", REGRESSION + EXPLAINABLE + ["Structured tabular data"], "Simple baseline for numeric targets with transparent coefficients.", "Relationships are roughly linear and explainability matters.", "Sensitive to outliers, leakage, and nonlinear effects.", "Labeled tabular data with continuous target.", "RMSE, MAE, R2, MAPE when appropriate.", "Start as a baseline; inspect residuals and feature leakage."),
    _spec("Ridge / Lasso Regression", "General ML / Structured Data", REGRESSION + EXPLAINABLE + ["High-cardinality categorical variables"], "Regularized regression handles many correlated features better than plain linear regression.", "You need a stable numeric baseline with many predictors.", "Still linear; categorical encoding must be leakage-safe.", "Continuous target with engineered features.", "RMSE, MAE, R2.", "Tune alpha with cross-validation; use pipelines for preprocessing."),
    _spec("Decision Tree", "General ML / Structured Data", CLASSIFICATION + REGRESSION + EXPLAINABLE, "Readable tree rules can model nonlinear splits.", "A transparent baseline or decision policy is needed.", "Single trees overfit easily and can be unstable.", "Labeled tabular data.", "F1, ROC-AUC, RMSE, MAE depending on target.", "Limit depth, export rules, and compare against ensembles."),
    _spec("Random Forest", "General ML / Structured Data", CLASSIFICATION + REGRESSION + ["Noisy inputs", "Missing values", "Outliers are expected"], "Robust ensemble for tabular classification or regression.", "You need a strong low-maintenance baseline.", "Less interpretable than simple models; large models can be slower.", "Labeled tabular data.", "F1, ROC-AUC, RMSE, MAE.", "Use permutation importance or SHAP; tune depth and class weights."),
    _spec("Extra Trees", "General ML / Structured Data", CLASSIFICATION + REGRESSION + ["Noisy inputs", "Large dataset"], "Highly randomized tree ensemble that can perform well on noisy tabular data.", "Fast ensemble experiments are needed.", "Can be less stable for small datasets.", "Labeled tabular data.", "F1, ROC-AUC, RMSE, MAE.", "Compare against Random Forest and gradient boosting."),
    _spec("Gradient Boosting", "General ML / Structured Data", CLASSIFICATION + REGRESSION + ["Need high accuracy even if complex"], "Strong tabular model family for nonlinear predictive tasks.", "Accuracy matters and features are mostly structured.", "Needs careful validation, tuning, and drift monitoring.", "Labeled tabular data.", "F1, ROC-AUC, PR-AUC, RMSE, MAE.", "Use early stopping and leakage-safe validation."),
    _spec("XGBoost", "General ML / Structured Data", CLASSIFICATION + REGRESSION + ["Large dataset", "Need high accuracy even if complex", "Labels are highly imbalanced"], "High-performing gradient boosting implementation for structured data.", "You need competitive accuracy on tabular data.", "Complexity and tuning effort are higher than baselines.", "Labeled structured data.", "F1, ROC-AUC, PR-AUC, RMSE, MAE.", "Tune learning rate, depth, regularization, and imbalance handling."),
    _spec("LightGBM", "General ML / Structured Data", CLASSIFICATION + REGRESSION + ["Large dataset", "Very large dataset", "High-cardinality categorical variables"], "Efficient gradient boosting for large tabular datasets.", "Scale and speed matter for structured data.", "Can overfit if validation is weak; categorical handling needs care.", "Labeled structured data.", "F1, ROC-AUC, PR-AUC, RMSE, MAE.", "Use early stopping and monitor drift in production."),
    _spec("CatBoost", "General ML / Structured Data", CLASSIFICATION + REGRESSION + ["High-cardinality categorical variables", "Inconsistent formats"], "Gradient boosting with strong categorical feature handling.", "Many categorical variables exist and leakage control matters.", "Can be slower; still requires validation discipline.", "Labeled data with categorical predictors.", "F1, ROC-AUC, RMSE, MAE.", "Use ordered boosting and native categorical features when possible."),
    _spec("Support Vector Machine", "General ML / Structured Data", CLASSIFICATION + REGRESSION + ["Small dataset", "Medium dataset"], "Effective margin-based model for smaller structured or text feature spaces.", "Dataset is not huge and boundaries may be nonlinear.", "Scaling is difficult for very large data; calibration may be needed.", "Labeled data with scaled features or TF-IDF.", "F1, ROC-AUC, RMSE, MAE.", "Standardize features; test linear and kernel variants."),
    _spec("k-Nearest Neighbors", "General ML / Structured Data", CLASSIFICATION + REGRESSION + ["Small dataset", "Semantic similarity"], "Instance-based baseline useful when similarity is meaningful.", "Feature distances are trustworthy and data is modest.", "Poor latency and weak high-dimensional behavior without embeddings.", "Scaled numeric features or embeddings.", "Accuracy, F1, RMSE, MAE.", "Use as baseline; consider ANN search for larger embedding sets."),
    _spec("Naive Bayes", "General ML / Structured Data", CLASSIFICATION + ["Text data", "Small dataset"], "Fast probabilistic baseline, especially for text classification.", "You need a quick, explainable benchmark.", "Feature independence assumptions limit ceiling.", "Labeled text or categorical features.", "Accuracy, F1, precision, recall.", "Pair with bag-of-words or TF-IDF for text."),
    _spec("K-Means", "Unsupervised / Pattern Discovery", ["Group similar records without labels", "No reliable labels are available", "Explicitly no labels available", "Need unsupervised exploration first", "Structured tabular data"], "Simple clustering baseline for segment discovery.", "Clusters are roughly spherical and k can be estimated.", "Requires choosing k and scaling features.", "Unlabeled numeric features or embeddings.", "Silhouette, Davies-Bouldin, cluster stability.", "Standardize features; profile clusters for business meaning."),
    _spec("Hierarchical Clustering", "Unsupervised / Pattern Discovery", ["Group similar records without labels", "Small dataset", "Document clustering"], "Reveals nested similarity structure without labels.", "You need interpretable cluster hierarchy.", "Does not scale well to very large datasets.", "Unlabeled feature vectors or embeddings.", "Silhouette, dendrogram stability.", "Use on samples or embeddings; inspect dendrogram cuts."),
    _spec("DBSCAN", "Unsupervised / Pattern Discovery", ["Group similar records without labels", "Outliers are expected", "Detect unusual or abnormal behavior"], "Density-based clustering that can mark noise points.", "Clusters have dense regions and outliers matter.", "Sensitive to distance scale and epsilon.", "Scaled numeric features or embeddings.", "Silhouette, noise rate, cluster stability.", "Tune eps/min_samples and validate with domain review."),
    _spec("HDBSCAN", "Unsupervised / Pattern Discovery", ["Group similar records without labels", "Outliers are expected", "No reliable labels are available", "Explicitly no labels available", "Document clustering"], "Density clustering with variable-density support and noise labels.", "Cluster count is unknown and data has irregular clusters.", "May produce many noise points; parameters need review.", "Numeric features or embeddings.", "Cluster persistence, noise rate, qualitative review.", "Works well with UMAP embeddings for exploration."),
    _spec("Gaussian Mixture Models", "Unsupervised / Pattern Discovery", ["Group similar records without labels", "Find patterns or recurring structures"], "Probabilistic clustering with soft membership.", "Overlapping clusters and membership probabilities are useful.", "Assumes Gaussian-like components.", "Scaled numeric features.", "BIC, AIC, silhouette, membership entropy.", "Use for soft segmentation and uncertainty."),
    _spec("PCA", "Unsupervised / Pattern Discovery", ["Find patterns or recurring structures", "High-cardinality categorical variables", "Need business-friendly explanation"], "Linear dimensionality reduction for compression and diagnostics.", "You need interpretable components or denoising.", "Linear method; components may not be business intuitive.", "Numeric feature matrix.", "Explained variance, reconstruction error.", "Use before clustering or visualization."),
    _spec("UMAP", "Unsupervised / Pattern Discovery", ["Find patterns or recurring structures", "Document clustering", "Visual similarity search"], "Nonlinear embedding useful for visualization and clustering.", "You need exploratory maps of high-dimensional data.", "Distances in 2D can be misleading.", "Embeddings or numeric feature vectors.", "Trustworthiness, cluster stability.", "Use with HDBSCAN and qualitative inspection."),
    _spec("t-SNE", "Unsupervised / Pattern Discovery", ["Find patterns or recurring structures", "Small dataset"], "Visualization method for local neighborhoods.", "You need exploratory visualization of embeddings.", "Not ideal for production features or global distances.", "Numeric vectors or embeddings.", "Qualitative neighborhood review.", "Use for visual exploration, not final clustering."),
    _spec("Association Rules", "Unsupervised / Pattern Discovery", ["Transactional data", "Find patterns or recurring structures", "Recommend items or actions"], "Discovers interpretable co-occurrence rules.", "Basket, event, or transaction patterns matter.", "Can produce many spurious rules.", "Transactional item sets.", "Support, confidence, lift.", "Filter by lift and minimum support; validate with business owners."),
    _spec("Frequent Pattern Mining", "Unsupervised / Pattern Discovery", ["Transactional data", "Find patterns or recurring structures"], "Finds recurring itemsets or event combinations.", "You need pattern discovery from transactions or logs.", "Combinatorial growth can be expensive.", "Transaction/event data.", "Support, coverage, lift.", "Use Apriori/FP-growth and strict support thresholds."),
    _spec("Isolation Forest", "Anomaly Detection", ["Detect unusual or abnormal behavior", "Outliers are expected", "No reliable labels are available", "Explicitly no labels available", "Structured tabular data"], "Tree-based unsupervised anomaly detector.", "Anomalies are rare and labels are unavailable.", "Scores need threshold calibration and review.", "Tabular features or embeddings.", "Precision@k, recall@k, PR-AUC if labels exist.", "Use human review to set thresholds."),
    _spec("One-Class SVM", "Anomaly Detection", ["Detect unusual or abnormal behavior", "Small dataset", "No reliable labels are available", "Explicitly no labels available"], "Learns boundary around normal examples.", "Mostly normal data is available and dataset is modest.", "Sensitive to scaling and hyperparameters.", "Normal-class feature vectors.", "Precision@k, recall@k, false alarm rate.", "Scale features and tune nu/gamma carefully."),
    _spec("Local Outlier Factor", "Anomaly Detection", ["Detect unusual or abnormal behavior", "Outliers are expected", "Small dataset"], "Detects local density deviations.", "Local neighborhood anomalies matter.", "Not ideal for very large or high-dimensional raw data.", "Scaled feature vectors.", "Precision@k, recall@k.", "Use for offline detection and compare to Isolation Forest."),
    _spec("Statistical Thresholds / Z-score", "Anomaly Detection", ["Detect unusual or abnormal behavior", "Need clear decision thresholds", "High explainability required"], "Transparent thresholding baseline.", "Signals are simple and auditability matters.", "Weak for complex multivariate anomalies.", "Numeric signals with historical ranges.", "False positive rate, recall, alert volume.", "Use robust thresholds and documented exception handling."),
    _spec("Robust covariance", "Anomaly Detection", ["Detect unusual or abnormal behavior", "Outliers are expected", "Small dataset"], "Multivariate anomaly detection using robust covariance estimates.", "Numeric variables are correlated and sample size is manageable.", "Assumes elliptical distributions.", "Numeric feature matrix.", "Precision@k, Mahalanobis distance review.", "Use after scaling and outlier analysis."),
    _spec("Autoencoders", "Anomaly Detection", ["Detect unusual or abnormal behavior", "Large dataset", "Need high accuracy even if complex", "Sensor data"], "Learns reconstruction of normal patterns for anomaly scoring.", "Large normal datasets and nonlinear patterns exist.", "Harder to explain; thresholds require validation.", "Mostly normal high-dimensional data.", "Reconstruction error, PR-AUC, recall@k.", "Use simple baselines first; monitor drift."),
    _spec("Time-series anomaly detection", "Anomaly Detection", ["Need anomaly detection over time", "Time series data", "Logs or event streams", "Sensor data"], "Detects abnormal temporal behavior instead of static outliers.", "Sequence context and seasonality matter.", "Needs backtesting and alert calibration.", "Timestamped observations.", "Precision@k, recall, alert rate, detection delay.", "Use rolling windows and production alert review."),
    _spec("Naive / Seasonal Naive Baseline", "Time Series", ["Forecast future values over time", "Seasonality exists", "Time series data"], "Essential forecasting baseline.", "You need a sanity check before complex models.", "Cannot use rich covariates or nonlinear drivers.", "Historical time series.", "MAE, RMSE, MAPE/sMAPE, MASE.", "Always benchmark against this before advanced models."),
    _spec("Moving Average", "Time Series", ["Forecast future values over time", "Noisy inputs", "Time series data"], "Simple smoothing baseline.", "Short-term smoothing is useful.", "Lags trend shifts and ignores seasonality.", "Historical time series.", "MAE, RMSE.", "Use rolling validation."),
    _spec("Exponential Smoothing", "Time Series", ["Forecast future values over time", "Trend exists", "Seasonality exists"], "Classical forecasting for trend and seasonality.", "Series is univariate and interpretable forecasts are needed.", "Limited external variable support.", "Regular time series.", "MAE, RMSE, MASE, prediction interval coverage.", "Try ETS variants with rolling backtests."),
    _spec("ARIMA", "Time Series", ["Forecast future values over time", "Time series data", "Trend exists"], "Classical autoregressive baseline.", "Autocorrelation drives the series.", "Stationarity assumptions and tuning can be brittle.", "Regular univariate time series.", "MAE, RMSE, AIC, BIC.", "Use differencing and residual diagnostics."),
    _spec("SARIMA", "Time Series", ["Forecast future values over time", "Seasonality exists", "Time series data"], "ARIMA extension for seasonal patterns.", "Clear seasonality exists in a regular series.", "Can be slow for many series.", "Regular seasonal time series.", "MAE, RMSE, MASE.", "Use rolling backtesting and residual checks."),
    _spec("SARIMAX", "Time Series", ["Forecast future values over time", "External variables available", "Seasonality exists"], "Classical forecasting with exogenous variables.", "External drivers are known for the forecast horizon.", "Future covariates must be available or forecasted.", "Time series plus exogenous variables.", "MAE, RMSE, MASE, coverage.", "Validate covariate availability and leakage."),
    _spec("Prophet-like decomposable forecasting", "Time Series", ["Forecast future values over time", "Seasonality exists", "Need business-friendly explanation"], "Decomposes trend, seasonality, and events for explainable forecasts.", "Business users need interpretable components.", "Can underperform tuned ML on complex series.", "Timestamped historical data and optional events.", "MAE, RMSE, MAPE, coverage.", "Use when calendar effects matter and explainability is useful."),
    _spec("Gradient Boosting with lag features", "Time Series", ["Forecast future values over time", "External variables available", "Multiple related time series", "Large dataset"], "Turns forecasting into supervised learning with lags and covariates.", "Many series or rich external variables exist.", "Feature leakage is a major risk.", "Historical target, lag features, known covariates.", "MAE, RMSE, MASE, backtest metrics.", "Build leakage-safe feature windows and rolling validation."),
    _spec("LSTM / GRU", "Time Series", ["Forecast future values over time", "Large dataset", "Multiple related time series", "Need high accuracy even if complex"], "Recurrent neural networks for temporal patterns.", "Large sequence datasets and nonlinear dependencies exist.", "Data hungry, less interpretable, harder to maintain.", "Many historical sequences.", "MAE, RMSE, coverage.", "Use after classical and boosting baselines."),
    _spec("Temporal CNN", "Time Series", ["Forecast future values over time", "Large dataset", "Sensor data", "Multiple related time series"], "Convolutional sequence model for temporal signals.", "Local temporal patterns matter at scale.", "Architecture and windowing choices matter.", "Windowed time series data.", "MAE, RMSE, event metrics.", "Useful for sensors and event streams."),
    _spec("Transformer-based forecasting", "Time Series", ["Forecast future values over time", "Very large dataset", "Multiple related time series", "Need high accuracy even if complex"], "Attention-based forecasting for large multi-series problems.", "Large datasets justify complex sequence models.", "High compute and operational complexity.", "Large historical time series corpus.", "MAE, RMSE, MASE, coverage.", "Use only after baselines prove insufficient."),
    _spec("Hierarchical forecasting", "Time Series", ["Forecast future values over time", "Multiple related time series", "Support executive decision-making"], "Reconciles forecasts across aggregation levels.", "Forecasts must add up across product, region, or business hierarchy.", "Requires clean hierarchy definitions.", "Related time series and hierarchy metadata.", "MAE/RMSE by level, reconciliation error.", "Use bottom-up, top-down, or optimal reconciliation."),
    _spec("Regex and rule-based extraction", "NLP / Text", ["Keyword/rule-based patterns", "Extract information from text", "High explainability required", "Sensitive information detection"], "Transparent extraction for stable patterns.", "Formats are predictable and auditability matters.", "Brittle to variation and OCR errors.", "Text or OCR output.", "Precision, recall, exact match.", "Use as baseline and validation layer."),
    _spec("TF-IDF + Logistic Regression", "NLP / Text", ["Text data", "Text classification", "Classify documents", "Reliable labeled data is available"], "Strong classical baseline for text classification.", "Labeled documents are available and speed matters.", "Limited semantic understanding.", "Labeled text documents.", "Accuracy, macro F1, PR-AUC.", "Use n-grams, class weights, and calibration."),
    _spec("TF-IDF + Linear SVM", "NLP / Text", ["Text data", "Text classification", "Classify documents", "Small dataset"], "High-performing sparse text classifier.", "Text labels exist and dataset is moderate.", "Probabilities require calibration.", "Labeled text.", "Accuracy, macro F1, precision, recall.", "Tune C and class weights."),
    _spec("Naive Bayes text classifier", "NLP / Text", ["Text data", "Text classification", "Small dataset", "Need fast implementation"], "Very fast baseline for text categories.", "You need a simple benchmark quickly.", "Often lower ceiling than SVM or transformers.", "Labeled text.", "Accuracy, F1.", "Use MultinomialNB with TF-IDF/count features."),
    _spec("Named Entity Recognition", "NLP / Text", ["Named entity recognition", "Detect entities or sensitive information", "Need structured extraction"], "Finds entities and spans in text.", "Entities must be extracted from unstructured text.", "Needs labeled spans or strong pretrained models.", "Text with entity labels or pretrained model.", "Entity F1, precision, recall.", "Combine with rules for sensitive data validation."),
    _spec("Transformer text classifier", "NLP / Text", ["Text data", "Text classification", "Need high accuracy even if complex", "Long documents"], "Pretrained language model fine-tuning for text classification.", "Semantics matter and enough labeled data exists.", "More compute, monitoring, and explainability burden.", "Labeled text; possibly chunked long docs.", "Macro F1, ROC-AUC, PR-AUC.", "Start from classical baselines; chunk long documents carefully."),
    _spec("Sentence embeddings", "NLP / Text", ["Semantic similarity", "Search semantically across documents", "Text data", "Need retrieval over documents"], "Dense semantic representations for search, clustering, and similarity.", "Meaning-based matching is needed.", "Embedding drift and domain mismatch can affect quality.", "Text chunks/documents.", "Recall@k, MRR, nDCG, clustering metrics.", "Choose domain-appropriate embeddings; evaluate retrieval."),
    _spec("Topic Modeling with LDA", "NLP / Text", ["Topic modeling", "Find patterns or recurring structures", "Text data"], "Classical unsupervised topic discovery.", "You need interpretable word-topic themes.", "Can be brittle and less semantic than embedding methods.", "Text corpus.", "Topic coherence, qualitative review.", "Use preprocessing and compare to BERTopic."),
    _spec("BERTopic", "NLP / Text", ["Topic modeling", "Document clustering", "Text data", "Semantic similarity"], "Embedding-based topic modeling.", "Semantic topics are needed without labels.", "Topic labels require review; embeddings matter.", "Text corpus and embeddings.", "Topic coherence, diversity, review quality.", "Use with UMAP/HDBSCAN and manual topic naming."),
    _spec("Semantic Search", "NLP / Text", ["Semantic similarity", "Search semantically across documents", "Need retrieval over documents"], "Searches by meaning instead of exact keywords.", "Users need to find relevant documents or passages.", "Needs chunking, embedding, and relevance evaluation.", "Documents split into chunks.", "Recall@k, MRR, nDCG.", "Use vector index and reranking for quality."),
    _spec("Text clustering with embeddings", "NLP / Text", ["Document clustering", "Text data", "No reliable labels are available", "Explicitly no labels available"], "Groups semantically similar documents.", "No labels exist and exploration is needed.", "Cluster interpretation requires human review.", "Text embeddings.", "Silhouette, cluster stability, review accuracy.", "Use HDBSCAN/K-Means over embeddings."),
    _spec("Summarization models", "NLP / Text", ["Need summarization", "Long documents", "Generate natural language answers"], "Condenses long text for review workflows.", "Users need concise document understanding.", "Can omit critical details; needs validation.", "Documents or chunks.", "ROUGE, factuality checks, human review.", "Use chunked summarization and citation checks."),
    _spec("Structured extraction with LLMs", "NLP / Text", ["Need structured extraction", "Extract information from text", "Long documents", "Scanned documents"], "LLMs can extract fields from varied unstructured text.", "Schemas are known but document language varies.", "Requires validation, guardrails, and review for high-risk fields.", "Text/OCR and target schema.", "Exact match, field F1, validation error rate.", "Use JSON schema validation and human review."),
    _spec("OCR + rules", "Document Intelligence", ["Scanned documents", "OCR output", "Keyword/rule-based patterns", "High explainability required"], "Combines OCR with transparent extraction rules.", "Documents have stable templates or known fields.", "OCR errors and template variation can break rules.", "Scanned pages or OCR text.", "Field precision/recall, OCR word error rate.", "Preprocess scans and log rule decisions."),
    _spec("OCR + NLP classifier", "Document Intelligence", ["Scanned documents", "OCR output", "Document classification", "Classify documents"], "Uses OCR text for document classification.", "Scanned documents need routing or categorization.", "OCR quality affects downstream performance.", "OCR text and document labels.", "Macro F1, accuracy, confusion matrix.", "Start with TF-IDF; upgrade to transformers if needed."),
    _spec("Layout-aware models", "Document Intelligence", ["Layout-aware document processing", "Layout information", "Scanned documents", "Understand document layout"], "Uses text positions and layout for document understanding.", "Field meaning depends on where text appears.", "Needs OCR coordinates and specialized pipelines.", "OCR tokens with bounding boxes/layout.", "Field F1, exact match, document F1.", "Use for forms, invoices, tables, and compliance docs."),
    _spec("LayoutLM-style document understanding", "Document Intelligence", ["Layout-aware document processing", "Spatial coordinates or bounding boxes", "Layout information"], "Transformer architecture combining text and layout.", "Token positions are critical for extraction/classification.", "Requires annotated examples and bounding boxes.", "OCR tokens, bounding boxes, labels.", "Entity F1, field exact match.", "Fine-tune from pretrained document models."),
    _spec("Donut-style OCR-free document understanding", "Document Intelligence", ["Scanned documents", "Need OCR plus vision", "Need high accuracy even if complex"], "Vision-to-text document model that can bypass separate OCR.", "OCR is poor or layout is visually complex.", "Data and compute requirements are higher.", "Document images and task labels.", "Exact match, normalized edit distance.", "Consider when OCR pipeline is limiting quality."),
    _spec("Document embeddings + clustering", "Document Intelligence", ["Document clustering", "Scanned documents", "Long documents", "No reliable labels are available", "Explicitly no labels available"], "Clusters documents using semantic embeddings.", "Unlabeled document collections need organization.", "OCR/chunking quality affects clusters.", "OCR/text chunks or document text.", "Silhouette, review quality.", "Embed chunks/documents and review clusters with SMEs."),
    _spec("Template matching", "Document Intelligence", ["Need template matching", "Scanned pages", "Form/table detection"], "Classical approach for stable visual templates.", "Document forms are highly standardized.", "Fragile when layouts change.", "Images/templates or anchor fields.", "Detection precision/recall.", "Use as a transparent baseline for fixed forms."),
    _spec("Metadata validation rules", "Document Intelligence", ["Need metadata validation", "Auditability required", "Compliance requirements apply"], "Validates extracted fields against business constraints.", "Compliance or handoff requires reliable checks.", "Rules must be maintained as policies change.", "Extracted fields and metadata.", "Validation pass rate, false reject rate.", "Use after OCR/LLM extraction as guardrails."),
    _spec("Hybrid rules + ML review decisioning", "Document Intelligence", ["Human-in-the-loop required", "Reduce manual review workload", "Automation must be conservative", "Auditability required"], "Combines deterministic rules, model scores, and human review routing.", "Risk is high and partial automation is acceptable.", "Needs workflow design and threshold governance.", "Predictions, confidence scores, review outcomes.", "Automation rate, precision, recall, override rate.", "Define auto-approve, auto-reject, and review bands."),
    _spec("Near-duplicate detection with MinHash", "Document Intelligence", ["Need deduplication or near-duplicate detection", "Text data", "Long documents"], "Efficient text near-duplicate detection.", "Documents are text-heavy and duplicates matter.", "Less useful for image-only duplicates.", "Tokenized text or shingles.", "Duplicate precision/recall.", "Use shingling + LSH for scale."),
    _spec("Perceptual hashing for scanned pages", "Document Intelligence", ["Scanned pages", "Need deduplication or near-duplicate detection", "Images"], "Finds visually similar scanned pages.", "Image-level duplicates or scan variants exist.", "Not semantic; sensitive to some transformations.", "Page images.", "Duplicate precision/recall.", "Use pHash/dHash and threshold calibration."),
    _spec("Image classification CNN", "Computer Vision", ["Whole image classification", "Images", "Predict a known category or class"], "Classifies entire images into categories.", "Labels exist at image level.", "Needs enough labeled images; may miss local objects.", "Labeled images.", "Accuracy, macro F1, ROC-AUC.", "Start with transfer learning."),
    _spec("Transfer learning with ResNet/EfficientNet", "Computer Vision", ["Images", "Small dataset", "Whole image classification", "Need fast implementation"], "Pretrained CNNs adapt well with limited image labels.", "Dataset is small to medium and image labels exist.", "Domain shift and augmentations matter.", "Labeled images.", "Accuracy, F1, confusion matrix.", "Freeze backbone first; fine-tune later."),
    _spec("Vision Transformer", "Computer Vision", ["Images", "Large dataset", "Need high accuracy even if complex"], "Transformer-based image model for high-capacity visual tasks.", "Large data or strong pretrained models are available.", "More compute and less transparent.", "Labeled images.", "Accuracy, F1.", "Use pretrained ViT when CNN baseline is insufficient."),
    _spec("YOLO object detection", "Computer Vision", ["Object detection required", "Detect objects or regions in images", "Low latency is required", "Small objects"], "Fast object detector for bounding boxes.", "Real-time or near real-time detection is needed.", "Small objects and low-light data need careful labeling and augmentation.", "Images with bounding boxes.", "mAP, precision, recall, FPS.", "Use YOLO for speed; tune anchors/img size if needed."),
    _spec("Faster R-CNN", "Computer Vision", ["Object detection required", "Region localization required", "Need high accuracy even if complex"], "Accurate two-stage object detector.", "Accuracy matters more than latency.", "Slower than one-stage detectors.", "Images with bounding boxes.", "mAP, precision, recall.", "Useful benchmark against YOLO."),
    _spec("Mask R-CNN", "Computer Vision", ["Pixel-level segmentation required", "Segment precise visual regions", "Object detection required"], "Instance segmentation with object masks.", "Objects need both detection and masks.", "Requires mask annotations.", "Images with instance masks.", "Mask mAP, IoU, Dice.", "Use when per-object masks matter."),
    _spec("U-Net segmentation", "Computer Vision", ["Pixel-level segmentation required", "Segment precise visual regions", "Images"], "Strong baseline for semantic segmentation.", "Pixel-level regions must be segmented.", "Requires mask labels; may need augmentation.", "Images and segmentation masks.", "IoU, Dice, pixel accuracy.", "Works well for medical, industrial, and document regions."),
    _spec("SAM-style segmentation", "Computer Vision", ["Pixel-level segmentation required", "Need high accuracy even if complex", "Need visual similarity search"], "Promptable segmentation useful for annotation and interactive workflows.", "You need flexible segmentation or annotation acceleration.", "May need domain adaptation and human validation.", "Images and optional prompts/masks.", "IoU, Dice, human correction time.", "Use to bootstrap masks or support review tools."),
    _spec("OCR + vision pipeline", "Computer Vision", ["Need OCR plus vision", "Scanned pages", "Form/table detection", "Sensitive region detection"], "Combines image preprocessing, OCR, and visual detection.", "Documents require both text and visual layout cues.", "Pipeline complexity and error propagation.", "Document images, OCR, boxes.", "Field F1, detection mAP, OCR WER.", "Use OpenCV preprocessing before OCR and layout models."),
    _spec("Image embeddings similarity search", "Computer Vision", ["Need visual similarity search", "Images", "Search semantically across documents"], "Finds visually similar images or pages.", "Similarity search or deduplication is needed.", "Embedding quality must match domain.", "Images and embedding index.", "Recall@k, MRR, human relevance.", "Use CLIP/vision embeddings and vector search."),
    _spec("Template matching / classical CV", "Computer Vision", ["Need template matching", "Scanned pages", "Signature detection", "Low-code/simple solution"], "Transparent computer vision for fixed visual patterns.", "Templates are stable and explainability matters.", "Fragile to rotation, lighting, and layout variation.", "Images and template regions.", "Precision, recall, localization IoU.", "Use OpenCV preprocessing and threshold review."),
    _spec("OpenCV preprocessing", "Computer Vision", ["Noisy images", "Low-light conditions", "Scanned pages", "OCR correction needed"], "Improves image quality before OCR or CV models.", "Noise, skew, contrast, or scan quality is an issue.", "Preprocessing can damage signal if overdone.", "Raw images/scans.", "OCR WER, downstream F1.", "Use thresholding, deskewing, denoising, contrast correction."),
    _spec("RAG pipeline", "GenAI / RAG / Agents", ["Need RAG", "Need retrieval over documents", "Generate natural language answers", "Need question answering over documents"], "Retrieves relevant context before generating answers.", "Answers must be grounded in documents.", "Needs evaluation against hallucination and retrieval misses.", "Document corpus, chunks, embeddings.", "Faithfulness, answer relevance, recall@k.", "Use citations, chunking, reranking, and guardrails."),
    _spec("Vector search with embeddings", "GenAI / RAG / Agents", ["Search semantically across documents", "Need retrieval over documents", "Semantic similarity"], "Core retrieval layer for semantic search and RAG.", "Semantic document lookup is required.", "Chunking and embedding choice drive quality.", "Chunked text/documents and vector index.", "Recall@k, MRR, nDCG.", "Evaluate retrieval separately before generation."),
    _spec("LLM structured extraction", "GenAI / RAG / Agents", ["Need structured extraction", "Extract information from text", "Long documents"], "Extracts structured fields from varied text using prompts/schema.", "Rules are too brittle but fields are known.", "Must validate outputs and handle uncertainty.", "Text/OCR and schema.", "Field F1, exact match, validation failures.", "Use schema validation and human-in-loop for low confidence."),
    _spec("LLM classifier with validation", "GenAI / RAG / Agents", ["Text classification", "Labels are scarce", "Need fast implementation", "Need business-friendly explanation"], "Prompted classifier can bootstrap labels or handle nuanced categories.", "Few labels exist and semantics matter.", "Cost, latency, drift, and consistency need controls.", "Text and clear label taxonomy.", "Agreement, F1 on review set, calibration.", "Use deterministic validation and benchmark against classical models."),
    _spec("Agentic workflow with tool use", "GenAI / RAG / Agents", ["Automate workflow decisions", "Integration with existing workflow required", "API endpoint required"], "LLM coordinates tools for multi-step workflows.", "Tasks require retrieval, validation, and actions.", "Higher operational risk; needs guardrails and logs.", "Tools/APIs, policies, test cases.", "Task success, error rate, intervention rate.", "Keep tools narrow and auditable."),
    _spec("Multi-agent workflow", "GenAI / RAG / Agents", ["Automate workflow decisions", "Team can support complex ML system", "Need high accuracy even if complex"], "Specialized agents can divide complex review tasks.", "Workflow has distinct roles and enough engineering support.", "Complexity can exceed benefit.", "Clear task decomposition and evaluation set.", "Task success, review quality, cost.", "Use only after single-agent/tool workflow is insufficient."),
    _spec("Function calling / tool calling", "GenAI / RAG / Agents", ["Automate workflow decisions", "API endpoint required", "Integration with existing workflow required"], "Structured tool calls connect LLM reasoning to systems.", "Model must request actions or validations.", "Tool schemas and permissions must be tightly controlled.", "Tool definitions, APIs, validation rules.", "Tool-call accuracy, task success.", "Log every call and validate arguments."),
    _spec("Guardrails and validation layer", "GenAI / RAG / Agents", ["Compliance requirements apply", "Auditability required", "Need fallback logic", "Need confidence scores"], "Validation layer reduces unsafe or invalid AI outputs.", "Risk, compliance, or structured outputs matter.", "Cannot fix poor upstream retrieval/model design alone.", "Policies, schemas, validators, thresholds.", "Violation rate, false block rate, override rate.", "Combine rules, schema checks, confidence, and review routing."),
    _spec("Human-in-the-loop AI workflow", "GenAI / RAG / Agents", ["Human-in-the-loop required", "Reduce manual review workload", "High cost of false positives", "High cost of false negatives"], "Routes uncertain or risky cases to humans.", "Automation must be conservative.", "Requires review UX and feedback capture.", "Predictions, confidence, review labels.", "Automation rate, precision, recall, override rate.", "Define thresholds and feedback loop from day one."),
    _spec("Prompt-based triage assistant", "GenAI / RAG / Agents", ["Reduce manual review workload", "Generate natural language answers", "Need fast implementation"], "LLM helps prioritize and summarize cases.", "Users need decision support rather than full automation.", "Needs clear limits and review process.", "Case text/documents and triage rubric.", "Reviewer time saved, agreement rate.", "Start as assistant with citations and confidence."),
    _spec("Rule-based decision engine", "Optimization / Decisioning", ["Automate workflow decisions", "High explainability required", "Need clear decision thresholds", "Auditability required"], "Transparent deterministic decisions.", "Rules are known and compliance matters.", "Can become brittle and hard to maintain.", "Business rules and input fields.", "Coverage, override rate, error rate.", "Version rules and log every decision."),
    _spec("Scoring model", "Optimization / Decisioning", ["Rank or score alternatives", "Need confidence scores", "Structured tabular data"], "Produces scores for prioritization or risk.", "Ranking or triage is the real business action.", "Scores need calibration and threshold governance.", "Historical outcomes and features.", "AUC, PR-AUC, calibration, lift.", "Use score bands and monitor distribution shift."),
    _spec("Linear programming", "Optimization / Decisioning", ["Optimize a process or resource allocation", "Need business-friendly explanation"], "Optimizes continuous decisions under linear constraints.", "Objective and constraints are linear.", "Cannot directly model integer choices or nonlinear effects.", "Objective coefficients, constraints, capacities.", "Objective value, constraint violations.", "Use scipy/pulp/ortools and sensitivity analysis."),
    _spec("Integer programming", "Optimization / Decisioning", ["Optimize a process or resource allocation", "Automation must be conservative", "Need scalable pipeline"], "Optimizes discrete assignments or schedules.", "Decisions are yes/no or integer quantities.", "Can be computationally hard at large scale.", "Decision variables, constraints, objective.", "Objective value, feasibility, solve time.", "Use OR-Tools/PuLP and fallback heuristics."),
    _spec("Constraint optimization", "Optimization / Decisioning", ["Optimize a process or resource allocation", "Need scalable pipeline", "Need clear decision thresholds"], "Finds feasible or optimal decisions under constraints.", "Rules and constraints dominate the problem.", "Modeling constraints correctly is the hard part.", "Constraints, objective, domain rules.", "Feasibility, objective value, violations.", "Start with a clear mathematical formulation."),
    _spec("Reinforcement learning", "Optimization / Decisioning", ["Optimize a process or resource allocation", "Need high accuracy even if complex", "Feedback loop required"], "Learns policies through sequential rewards.", "A simulator or safe experimentation environment exists.", "Usually inappropriate without abundant interaction data and safety controls.", "State, action, reward, simulator/logs.", "Cumulative reward, regret, policy safety.", "Treat as advanced research path after simpler optimization."),
    _spec("Multi-armed bandit", "Optimization / Decisioning", ["Recommend items or actions", "Feedback loop required", "Data arrives continuously"], "Balances exploration and exploitation for recommendations/actions.", "Online feedback is available and actions can be tested safely.", "Requires experimentation governance.", "Action logs and reward feedback.", "Regret, conversion, reward, guardrail metrics.", "Use for controlled online optimization."),
    _spec("Simulation-based optimization", "Optimization / Decisioning", ["Optimize a process or resource allocation", "Need scenario forecasting", "Support executive decision-making", "Need causal interpretation"], "Tests decisions through simulated scenarios.", "Real-world experimentation is costly or risky.", "Simulation assumptions can dominate outcomes.", "Process model, scenarios, distributions.", "Objective value, service level, risk metrics.", "Use for what-if analysis and robust decisions."),
]


def build_project_profile(
    project_name: str,
    description: str,
    domain: str,
    notes: str,
    selected_options: dict[str, list[str]],
) -> ProjectProfile:
    profile = ProjectProfile(project_name, description, domain, notes, selected_options)
    selected = profile.selected_labels()

    if not selected:
        profile.warnings.append("Please select at least one known project characteristic.")
        return profile

    if not (
        set(CLASSIFICATION + REGRESSION) & selected
        or selected.intersection({"No reliable labels are available", "Explicitly no labels available"})
    ):
        profile.warnings.append("Target and label availability are unclear; unchecked boxes are treated as unknown, not negative.")
    if "Production deployment required" in selected and "Model monitoring required" not in selected:
        profile.warnings.append("Production is selected, but monitoring requirements are not fully specified.")
    if "High cost of false positives" in selected and "High cost of false negatives" in selected:
        profile.warnings.append("Both false positives and false negatives are costly; threshold strategy needs explicit business tradeoffs.")
    if "Forecast future values over time" in selected and "Time series data" not in selected:
        profile.warnings.append("Forecasting objective selected, but time series data was not explicitly selected.")
    if ("Need RAG" in selected or "Generate natural language answers" in selected) and "Need retrieval over documents" not in selected:
        profile.warnings.append("GenAI answer generation may need retrieval/citation details to reduce hallucination risk.")

    return profile


def recommend_algorithms(profile: ProjectProfile) -> list[AlgorithmRecommendation]:
    selected = profile.selected_labels()
    if not selected:
        return []

    recommendations: list[AlgorithmRecommendation] = []
    for spec in CATALOG:
        matches = selected.intersection(spec.triggers)
        if not matches:
            continue

        score = spec.base_score + (len(matches) * 9)

        if selected.intersection(EXPLAINABLE) and spec.category in {
            "General ML / Structured Data",
            "Document Intelligence",
            "Optimization / Decisioning",
        }:
            if any(token in spec.name for token in ["Regression", "Decision Tree", "Rule", "Threshold", "Metadata", "Hybrid"]):
                score += 10
        if selected.intersection(PRODUCTION):
            score += 4
        if "Small dataset" in selected and spec.category in {"Computer Vision", "Time Series", "GenAI / RAG / Agents"}:
            if "Transfer learning" not in spec.name and "baseline" not in spec.notes.lower():
                score -= 10
        if "Team needs low-code/simple solution" in selected:
            if any(token in spec.name for token in ["Regex", "Rule", "Naive", "Linear", "Logistic", "Moving Average", "TF-IDF"]):
                score += 12
            if any(token in spec.name for token in ["Transformer", "LSTM", "Agentic", "Multi-agent", "Reinforcement"]):
                score -= 12
        if "Need high accuracy even if complex" in selected:
            if any(token in spec.name for token in ["XGBoost", "LightGBM", "CatBoost", "Transformer", "YOLO", "Mask R-CNN", "LayoutLM", "RAG"]):
                score += 12
        if "High cost of false negatives" in selected:
            score += 3
        if "High cost of false positives" in selected:
            score += 3

        score = max(0, min(100, score))
        if score < 20:
            continue

        cautions = spec.cautions
        if "High cost of false negatives" in selected:
            cautions += " Use recall-oriented thresholding and review missed-case risk."
        if "High cost of false positives" in selected:
            cautions += " Use precision-oriented thresholding and control review burden."
        if selected.intersection(PRODUCTION):
            cautions += " Production use needs monitoring, versioning, drift checks, and rollback plans."

        recommendations.append(
            AlgorithmRecommendation(
                name=spec.name,
                category=spec.category,
                why_applicable=f"{spec.why} Matched: {', '.join(sorted(matches))}.",
                best_when=spec.best_when,
                cautions=cautions,
                required_data=spec.required_data,
                typical_metrics=spec.metrics,
                implementation_notes=spec.notes,
                confidence_score=score,
            )
        )

    unique: dict[str, AlgorithmRecommendation] = {}
    for rec in recommendations:
        if rec.name not in unique or rec.confidence_score > unique[rec.name].confidence_score:
            unique[rec.name] = rec

    return sorted(unique.values(), key=lambda item: (-item.confidence_score, item.category, item.name))
