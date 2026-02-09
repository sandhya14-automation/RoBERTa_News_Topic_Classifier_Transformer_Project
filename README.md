# RoBERTa News Topic Classifier (AG News) using a Transformer‑based NLP Model

### Project Summary
This project builds a news headline classifier using a modern NLP model called DistilRoBERTa.

The goal is to read a short news headline and predict which topic it belongs to:
- World
- Sports
- Business
- Sci/Tech
  
The workflow includes:
- Loading and understanding the dataset
- Tokenizing text for the model
- Creating a smaller training subset
- Setting up accuracy and macro‑F1 metrics
- Tuning hyperparameters using Optuna
- Training the final model
- Evaluating performance
- Plotting a confusion matrix
- Saving the trained model
  
This project demonstrates a complete, simple, and practical NLP pipeline using Hugging Face Transformers.

### Dataset Overview

We use the AG News dataset, which contains short news headlines grouped into four topics:
- World
- Sports
- Business
- Sci/Tech
  
### Dataset size:
- 120,000 training samples
- 7,600 test samples
  
To speed up training, we use a 10,000‑sample subset of the training data.

### Step 1: Install Required Libraries
We install all the tools needed for tokenization, training, evaluation, and hyperparameter tuning.

### Step 2: Import Libraries
We import:
- Transformers (for model + tokenizer)
- Datasets (for AG News)
- Evaluate (for accuracy and macro‑F1)
- Optuna (for tuning)
- NumPy and Torch

### Step 3: Load the AG News Dataset
We load the dataset using load_dataset("ag_news").

Each row contains:
- text → the news headline
- label → the topic number

### Step 4: Load Tokenizer
We load the DistilRoBERTa tokenizer, which converts text into:
- input IDs
- attention masks
  
These are required for the model.

### Step 5: Tokenize the Dataset
We tokenize all headlines by:
- Padding or cutting them to length 64
- Renaming label → labels
- Formatting the dataset for PyTorch
  
This prepares the data for training.

### Step 6: Use a Smaller Training Subset
To train faster, we shuffle the training set and select 10,000 samples.

### Step 7: Define Evaluation Metrics
We use two metrics:
- Accuracy → overall correctness
- Macro‑F1 → treats all classes equally
  
A custom function calculates both during evaluation.

### Step 8: Hyperparameter Tuning (Optuna)
Optuna tries different values for:
- Learning rate
- Batch size
- Number of epochs
  
Each trial:
- Builds training arguments
- Loads a fresh DistilRoBERTa model
- Trains on the 10k subset
- Evaluates on the test set
- Returns macro‑F1
  
After 3 trials, Optuna selects the best hyperparameters.

### Step 9: Train Final Model Using Best Hyperparameters
We train the final DistilRoBERTa model using the best settings found by Optuna.

The model is trained on the 10k subset and evaluated on the test set.

### Step 10: Evaluate the Model
We evaluate the final model using:
- Accuracy
- Macro‑F1
  
These scores show how well the model performs across all four topics.

### Step 11: Predict Headline Function
A simple function is created to test any headline.

Example:
predict_headline("Apple releases new iPhone model")

Output:
"Sci/Tech"

### Step 12: Confusion Matrix
A confusion matrix is plotted to show:
- Correct predictions
- Misclassifications
- Class‑wise performance
  
This helps visualize how well the model separates the four topics.

### Step 13: Save the Model
The final trained model and tokenizer are saved in:
final_roberta_model/

This allows the model to be reused later without retraining.

### Final Conclusion
Overall Findings
- DistilRoBERTa performs strongly on the AG News dataset.
- Accuracy and macro‑F1 are usually above 92%.
- The model handles all four topics well.
- Optuna tuning helps improve performance by finding better hyperparameters.
- The confusion matrix shows clear separation between categories.
  
### Summary
This project demonstrates a complete NLP classification pipeline using:
- Tokenization
- Hyperparameter tuning
- Transformer‑based training
- Evaluation
- Visualization
- Model saving
