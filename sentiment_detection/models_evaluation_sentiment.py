#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd

# process io_data_df
io_data_df = (r"C:\Users\20245179\OneDrive - TU Eindhoven\LLM_EngD_project\Intent recognition Comments\annotate_data_subset\comments_sustainability_4labeling_lbld_io.csv")
rows = []
with open(io_data_df, encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',')
        text = ','.join(parts[:-4])  # Join all parts except last 4 as 'text'
        nu_words, lbl1, lbl2 = parts[-4:-1]
        rows.append({'text': text, 'nu_words': nu_words, 'lbl1': lbl1, 'lbl2': lbl2})

import pandas as pd
io_data_df = pd.DataFrame(rows)

io_data_df = io_data_df.iloc[1:].reset_index(drop=True)
import pandas as pd

# New column: value after '_', or empty string if no '_'
io_data_df['sentiment1'] = io_data_df['lbl1'].apply(lambda x: x.split('_')[1] if '_' in x else '')

# Update lbl1: keep value before '_' if present, else keep original
io_data_df['lbl1'] = io_data_df['lbl1'].apply(lambda x: x.split('_')[0] if '_' in x else x)

# all the rows that hhave "Criticism" in lbl1 mark senbtiment1 as "-1"
io_data_df.loc[io_data_df['lbl1'] == 'criticism', 'sentiment1'] = '-1'
# all the rows that have "Appreciation" in lbl1 mark sentiment1 as "1"
io_data_df.loc[io_data_df['lbl1'] == 'appreciation', 'sentiment1'] = '1'
# all the rows that have "inquiry" in lbl1 mark sentiment1 as "0"
io_data_df.loc[io_data_df['lbl1'] == 'inquiry', 'sentiment1'] = '0'
# all the rows that have "forward" in lbl1 mark sentiment1 as "0"
io_data_df.loc[io_data_df['lbl1'] == 'forward', 'sentiment1'] = '0'

#reepalce "n" in sentiment 1 as "0"
io_data_df['sentiment1'] = io_data_df['sentiment1'].replace('n', '0')
#replce "p" in sentiment1 as "1"
io_data_df['sentiment1'] = io_data_df['sentiment1'].replace('p', '1')






# In[15]:


io_data_df["sentiment1"].value_counts()

# In[6]:


#### process calrlijn_data
import pandas as pd
carlijn_data = pd.read_excel(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\sentiment pred\test_sentiment_data_262_comments_labeled.xlsx")
carlijn_data["PI_Sentiment"] = carlijn_data["PI_Sentiment"].str.lower()
carlijn_data["PI_Sentiment"] = carlijn_data["PI_Sentiment"].replace("neutral", "0")
carlijn_data["PI_Sentiment"] = carlijn_data["PI_Sentiment"].replace("positive", "1")
carlijn_data["PI_Sentiment"] = carlijn_data["PI_Sentiment"].replace("negative", "-1")

# In[7]:


carlijn_data

# In[8]:


carlijn_data_sent = carlijn_data[["Comments", "PI_Sentiment"]]

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
# Load dataset with 'text.y' and 'reewalk_label' columns
df = carlijn_data_sent
df = df[df["Comments"].notnull()].reset_index(drop=True)
df["PI_Sentiment"] = df["PI_Sentiment"].astype(int)

# Sentiment text to numeric label mapping
def to_numeric_label(label: str):
    label = label.lower()
    if label in ["positive", "pos", "label_1", "4 stars", "5 stars", "1", "very positive"]:
        return 1
    elif label in ["negative", "neg", "label_0", "1 star", "2 stars", "-1", "very negative"]:
        return -1
    else:
        return 0


# List of models to evaluate
models = [
    'DTAI-KULeuven/robbert-v2-dutch-sentiment',
    'pdelobelle/robbert-v2-dutch-base',
    'DTAI-KULeuven/robbertje-merged-dutch-sentiment',
    'nlptown/bert-base-multilingual-uncased-sentiment',
    'GroNLP/bert-base-dutch-cased',
    'citizenlab/twitter-xlm-roberta-base-sentiment-finetunned',
    "BramVanroy/xlm-roberta-base-hebban-reviews",
    "BramVanroy/bert-base-multilingual-cased-hebban-reviews",
    "BramVanroy/robbert-v2-dutch-base-hebban-reviews",
    "clips/republic",
]

results = {}

# Loop through models
for model_name in models:
    try:
        print(f"\n🔍 Loading model: {model_name}")
        classifier = pipeline("sentiment-analysis", model=model_name, return_all_scores=True, truncation=True, max_length=512)

        model_preds = []
        for text in df["Comments"]:
            try:
                scores = classifier(text)[0]
                if len(scores) == 2:
                    pos_score = scores[1]['score']
                    neg_score = scores[0]['score']
                    if abs(pos_score - neg_score) <= 0.1:
                        sentiment = "neutral"
                    elif pos_score > neg_score:
                        sentiment = "positive"
                    else:
                        sentiment = "negative"
                    model_preds.append(to_numeric_label(sentiment))

                else:
                    top = max(scores, key=lambda x: x["score"])
                    model_preds.append(to_numeric_label(top["label"]))
            except Exception as e:
                model_preds.append(0)

        df[model_name] = model_preds
        acc = accuracy_score(df["PI_Sentiment"], model_preds)
        print(f"✅ Accuracy: {acc:.4f}")
        print(classification_report(df["PI_Sentiment"], model_preds, digits=3))
        results[model_name] = acc
    

    except Exception as e:
        print(f"❌ Failed on {model_name}: {e}")

# Accuracy Summary
print("\n📊 Accuracy Summary:")
for model_name, acc in results.items():
    print(f"{model_name:65s} => {acc:.4f}")


# In[13]:


for model_name, acc in results.items():
    print(f"{model_name:65s} => {acc:.4f}")

# In[15]:


results

# In[19]:


#### evaluation of top 3 perforingg models using majprty voting



import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
from transformers import pipeline
import nltk
import pandas as pd
from collections import Counter
from tqdm import tqdm

nltk.download('punkt')

# Load the data
data =  carlijn_data_sent
data.replace('\n', pd.NA, inplace=True)

# Drop rows containing NaN values
#data.dropna(inplace=True)

# Reset index if needed
data.reset_index(drop=True, inplace=True)
print(len(data["Comments"]))

class zero_shot_voting():
    def __init__(self):
        # Initializing models for zero-shot sentiment classification
        self.models = [
            #'DTAI-KULeuven/robbert-v2-dutch-sentiment',  # Label Negative Positive
            #'DTAI-KULeuven/robbertje-merged-dutch-sentiment', # Negative or Positive
            'nlptown/bert-base-multilingual-uncased-sentiment', # Other models can be added
            "clips/republic",
            #"tabularisai/multilingual-sentiment-analysis"
            "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"
        ]
        self.classifiers = [pipeline(task="sentiment-analysis", model=model, return_all_scores=True, truncation=True, max_length=512) for model in self.models]

    def sentiment(self, prediction):
        # Converting all predictions into numerical format for consistency
        if(prediction=="Very Positive" or prediction=="Positive" or prediction == "LABEL_1" or prediction== '1' or prediction=="pos" or prediction=="4 stars" or prediction=="5 stars" or prediction=="positive"):
            return int(1)
        elif(prediction=="Very Negative" or prediction=="Negative" or prediction == "LABEL_0" or prediction=='-1' or prediction=="neg" or prediction=="1 star" or prediction=="negative" or prediction=="2 stars"):
            return int(-1)
        else:
            return int(0)

    def prediction(self, text):
        # Prediction from all models
        predictions = []
        prediction_scores = []
        for classifier in self.classifiers:
            result = classifier(text)[0]
            

            if(len(result) == 2):
                # Extract scores for positive and negative labels
                positive_score = result[1]['score']
                negative_score = result[0]['score']

                # Determine sentiment label based on scores
                if abs(positive_score - negative_score) <= 0.1:
                    sentiment_label = 'neutral'
                elif positive_score > negative_score:
                    sentiment_label = 'Positive'
                else:
                    sentiment_label = 'Negative'


                predictions.append(self.sentiment(sentiment_label))

            else:
                predictions.append(self.sentiment(max(result, key=lambda x: x['score'])['label']))
                prediction_scores.append(max(result, key=lambda x: x['score'])['score'])
        return predictions, prediction_scores

    def are_all_same(self, pred):
        return all(x == pred[0] for x in pred)

    def overall_sentiment_sentence(self, predictions):
        """
        Returns overall sentiment based on strict majority voting.
        If a label appears 2 or more times among model predictions, return that label.
        Else, return 0 (Neutral).
        """
        label_counts = Counter(predictions)

        for label, count in label_counts.items():
            if count >= 2:
                return label  # Strong majority found

        return 0  # Neutral if no label appears 2 or more times


    def sentiment_over_comment_majority(self, sentiment):
        if not sentiment:
            return 0  # Default to neutral if no sentiments

        counts = Counter(sentiment)
        most_common = max(counts.values(), default=0)  # Use default to avoid empty sequence error
        max_labels = [label for label, count in counts.items() if count == most_common]

        if len(max_labels) == 1:
            return max_labels[0]

        elif len(max_labels) == 2 and counts[max_labels[0]] == counts[max_labels[1]]:
            return 0  # Mixed sentiment case

        return 2  # Neutral for other ambiguous cases

    def sentiment_across_sentences(self, text):
        sentences = nltk.sent_tokenize(text, language='dutch')
        majority_voting_sentences = []
        overall_sentiment = []
        overall_sentiment_score = []
        for sentence in sentences:
            overall_sentiment1, overall_sentiment_score1 = self.prediction(sentence)
            overall_sentiment.append(overall_sentiment1)
            overall_sentiment_score.append(overall_sentiment_score1)
            majority_score = self.overall_sentiment_sentence(overall_sentiment1)
            majority_voting_sentences.append(majority_score)
        sentiment_over_comment = self.sentiment_over_comment_majority(majority_voting_sentences)
        return overall_sentiment, overall_sentiment_score, majority_voting_sentences, sentiment_over_comment


# DataFrame to store results
df = pd.DataFrame(columns=["Comment", "Sentiment of the four models", "Score", "Overall voted sentiment", "Overall Sentiment over comment from sentences", 
                           "Sentiment from models", "Sentiment from models score", "Sentiment over comment from models", "Score for entire comment"])

obj = zero_shot_voting()

# Process each comment
for i in tqdm(range(len(data["Comments"]))):
    overall_sentiment, overall_sentiment_score, overall_sentiment_voting, sentiment_over_comment = obj.sentiment_across_sentences(data["Comments"][i])

    # Sentiment over sending entire comment to models
    pred, pred_score = obj.prediction(data["Comments"][i])
    check = {
        
        'Comment': data["Comments"][i],
        'Sentiment of the four models': overall_sentiment,
        'Score': overall_sentiment_score,
        'Overall voted sentiment': overall_sentiment_voting,
        'Overall Sentiment over comment from sentences': sentiment_over_comment,
        "Sentiment from models": pred,
        "Sentiment from models score": pred_score,
        "Sentiment over comment from models": obj.overall_sentiment_sentence(pred),
        "Score for entire comment": pred_score
    }
    check = pd.DataFrame([check])
    df = pd.concat([df, check], ignore_index=True)





# In[20]:


df

# In[21]:


to_check = pd.DataFrame()
to_check["Comment"] = data["Comments"]
to_check["pred_sentiment"] = df["Sentiment over comment from models"]
to_check["real_sentiment"] = data["PI_Sentiment"]

# In[22]:


to_check

# In[23]:


y_true = [str(label) for label in to_check["real_sentiment"]]
y_pred = [str(label) for label in to_check["pred_sentiment"]]
import numpy as np
print(classification_report(y_true, y_pred, digits=3))

# In[ ]:



