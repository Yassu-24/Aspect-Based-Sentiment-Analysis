# Aspect-Based Sentiment Analysis on Hindi Movie Reviews

This project performs **Aspect-Based Sentiment Analysis (ABSA)** on Hindi movie reviews, leveraging multiple natural language processing (NLP) techniques and state-of-the-art machine learning models. ABSA focuses on extracting the sentiment concerning specific aspects of movie reviews, such as acting, direction, music, etc.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [POS Tagging](#pos-tagging)
4. [Named Entity Recognition (NER)](#named-entity-recognition-ner)
5. [Models Used](#models-used)
6. [Acknowledgments](#acknowledgments)

## Project Overview

This project addresses the challenge of performing **aspect-based sentiment analysis** on Hindi movie reviews, which involves:
- Extracting important aspects (e.g., acting, direction, music) from the review text.
- Analyzing the sentiment (positive, negative, or neutral) associated with each aspect.

We utilize various machine learning techniques and state-of-the-art language models to improve the performance and accuracy of the sentiment analysis.

## Data Preprocessing

The dataset consists of Hindi movie reviews, which were preprocessed to remove unwanted noise, handle missing values, and tokenize the text for further analysis.

Key steps in preprocessing include:
- **Tokenization**: Dividing the text into words and phrases.
- **Stopword Removal**: Removing common words (e.g., "और", "है") that do not add significant value.
- **Stemming and Lemmatization**: Reducing words to their base form to ensure consistency in feature extraction.

## POS Tagging

To better understand the grammatical structure of the sentences, we performed **Part-of-Speech (POS) Tagging** using two methods:
1. **Conditional Random Fields (CRF)**: A probabilistic sequence model that assigns POS tags based on context.
2. **HMM-Viterbi Algorithm**: A Hidden Markov Model (HMM) approach combined with the Viterbi algorithm to assign the most probable sequence of POS tags.

These methods help in extracting useful grammatical features that improve the performance of subsequent models.

## Named Entity Recognition (NER)

For identifying and categorizing named entities (e.g., actor names, movie titles), we used **Polyglot** for NER. This helps in understanding the context of the review and extracting movie-specific information.

## Models Used

### 1. **Large Language Models (LLM)**
   - Utilized to capture contextual meanings in the reviews and enhance the understanding of sentiment related to specific aspects.

### 2. **BERT (Bidirectional Encoder Representations from Transformers)**
   - Fine-tuned a pre-trained BERT model for Hindi text to capture the context and meaning of the reviews. BERT’s attention mechanism helps in understanding long-range dependencies in the text.

### 3. **DistilBERT**
   - A smaller, faster, and lighter version of BERT that retains much of BERT's language understanding capabilities while being more efficient for processing.

### 4. **RoBERTa**
   - A robustly optimized BERT model that improves upon BERT by using larger batches and more data, enhancing performance in sentiment analysis tasks.

## Acknowledgments

This project was inspired by my passion for combining **Natural Language Processing (NLP)** with machine learning to solve real-world challenges in analyzing Indian language texts.

I would like to acknowledge the use of the following libraries:
- **spaCy** for POS tagging and text preprocessing.
- **Polyglot** for Named Entity Recognition.
- **Hugging Face Transformers** for pre-trained models like BERT, DistilBERT, and RoBERTa.
