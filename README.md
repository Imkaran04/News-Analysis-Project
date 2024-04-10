# News-Analysis-Project

## Program Execution Guide

Tools/Libraries Used:

1. pandas: For data manipulation and analysis.

2. scikit-learn (sklearn): For TF-IDF vectorization and Latent Dirichlet Allocation (LDA) topic modeling.

3. nltk (Natural Language Toolkit): For sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).

4. spacy: For text preprocessing and named entity recognition.

Steps to Run the Program:

1. Install Required Libraries: Make sure you have the necessary libraries installed. You can install them using pip:

pip install pandas scikit-learn nltk spacy

Additionally, download the English language model for spaCy:

python -m spacy download en_core_web_sm

2. Data Loading: The program assumes that the input data file is located at /kaggle/input/news-articles/Assignment.txt. If your file is located elsewhere or has a different name, modify the file path accordingly. The data file should contain only one column with articles.

3. Run the Code: Copy and paste the provided code into a Python script or Jupyter Notebook.

4. Execution: Execute the code in your Python environment. This can be done by running the script or executing each cell in the Jupyter Notebook.

5. Output: After running the program, the results will be saved to a new CSV file named result_with_txt.csv. This file will contain columns for the original articles, cleaned articles, sentiment analysis, identified topics, and aspects.

6. Interpretation: You can analyze the generated topics and aspects in the output CSV file to gain insights into the content of the articles. Additionally, review the sentiment analysis to understand the overall sentiment conveyed in the articles.

Why These Libraries:

1. pandas: Used for data manipulation and handling CSV files efficiently.

2. scikit-learn (sklearn): Utilized for TF-IDF vectorization and Latent Dirichlet Allocation (LDA) topic modeling due to its extensive machine learning functionality and ease of use.

3. nltk: Employed for sentiment analysis using VADER due to its comprehensive set of natural language processing tools.

4. spacy: Chosen for text preprocessing and named entity recognition for its fast and efficient performance and its availability of pre-trained models.

Why These Import Statements:

import pandas as pd: Pandas is imported with an alias pd for easier referencing throughout the code.

from sklearn.feature_extraction.text import TfidfVectorizer: Imports the TF-IDF vectorizer from scikit-learn for converting text data into numerical vectors.

from sklearn.decomposition import LatentDirichletAllocation: Imports the LDA model from scikit-learn for topic modeling.

from nltk.sentiment.vader import SentimentIntensityAnalyzer: Imports the VADER sentiment analysis tool from NLTK for analyzing sentiment in text data.

import spacy: Imports the spaCy library for advanced text processing tasks such as lemmatization and named entity recognition.




