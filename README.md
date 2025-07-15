# Text Classification in Spanish using BERT
## Project Overview

This project implements a **text classification system using BERT** (Bidirectional Encoder Representations from Transformers) to categorize Spanish user messages into three categories:

Information Requests
Complaints
Recommendations
The goal is to demonstrate the potential of **pre-trained language models for Spanish-language tasks**, covering the full pipeline: data preparation, model training, evaluation, and error analysis — all with a focus on real-world applicability.

## Key Features

- Fine-tunes the Spanish BERT model: dccuchile/bert-base-spanish-wwm-uncased
- Implements a three-class classification task
- Includes synthetic and augmented training data
- Provides detailed performance evaluation and visualizations
- Tests model robustness with a curated set of challenging, real-like examples

## Why It Matters

Efficient classification of user messages is essential for:

- Improving customer support workflows
- Detecting common pain points and user trends
- Prioritizing messages in high-volume scenarios
- This project showcases how state-of-the-art models can help interpret messages in Spanish, a language still underrepresented in many NLP applications.

## Dataset

- The dataset is embedded directly in the notebook.
- It consists of short messages in Spanish labeled into three categories:
   - 0 = Information Request
   - 1 = Complaint
   - 2 = Recommendation
- Each class contains approximately 100 synthetic examples, created to reflect realistic user intents.
- A separate set of challenging examples was added to test the model’s ability to deal with ambiguity, mixed intent, and subtle tone.

## Technologies Used

- Python
- Jupyter Notebook
- pandas, scikit-learn for data manipulation and metrics
- SimpleTransformers (built on HuggingFace Transformers)
- Matplotlib & Seaborn for visualizations

## Installation and Usage

### Option 1: Run locally

git clone https://github.com/BeaEsparcia/Spanish_Text_Classification_BERT.git
cd Spanish_Text_Classification_BERT
pip install -r requirements.txt
jupyter notebook

Then open bert_spanish_intent_classifier.ipynb and run all cells.

### Option 2: Run on Google Colab

1. Open Colab and upload the notebook.
2. Go to Runtime > Change runtime type and select GPU.
3. Run all cells.
4. If needed, run:
   !pip install simpletransformers pandas scikit-learn matplotlib seaborn

## Methodology

### Preprocessing & Data Augmentation

- Messages were written in natural, conversational Spanish.
- Synthetic, class-balanced dataset (≈100 examples per class).
- Examples were designed to simulate real user messages across three categories.
- A curated set of challenging messages — including multi-intent and ambiguous phrasing — was created to test model robustness.

### Model Selection

- Initial tests with an English BERT model performed poorly on Spanish.
- Switching to dccuchile/bert-base-spanish-wwm-uncased significantly improved results.

### Training Process

- Trained using **SimpleTransformers** with:
   - Early stopping
   - Evaluation after each epoch
   - Weight decay for regularization

### Evaluation

- Used classification_report and confusion matrix.
- Metrics: Precision, Recall, F1-score (macro and per-class)
- Heatmap visualization for interpretability

## Results & Observations

- Perfect performance on the clean validation set
- 47% accuracy on synthetic challenging examples designed to simulate real-world ambiguity.

### Common failure cases:

- Requests expressed as complaints
- Messages with mixed tones (e.g. “great product, but…”)
- Ambiguous language or multi-intent phrasing

## Limitations & Future Work

- Expand the dataset with real-world messages from users
- Explore advanced preprocessing (e.g. semantic clustering, negation handling)
- Test other architectures like RoBERTa, DistilBERT, or mDeBERTa
- Implement intent detection with multi-label classification or confidence thresholds

## What I Learned

- Model choice matters: a Spanish-specific model was essential for good performance.
- Clean data ≠ real performance: challenging examples revealed weaknesses hidden by high scores.
- Spanish NLP needs more attention: toolkits and models are still largely English-focused.
  
## Contributions

Open to suggestions and improvements!
Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See LICENSE.md for more information.

## Contact

Beatriz Esparcia - esparcia.beatriz@gmail.com
LinkedIn: www.linkedin.com/in/beaesparcia



   





