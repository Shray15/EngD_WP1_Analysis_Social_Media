import pandas as pd

# =============================================================================
# CONFIGURATION — update these paths for your environment
# =============================================================================
SENTIMENT_CSV = r"PATH_TO_SENTIMENT_CSV"   # e.g. results/Sentiments_cleaned_comments.csv
INTENT_CSV    = r"PATH_TO_INTENT_CSV"      # e.g. results/intent_predictions.csv
OUTPUT_CSV    = "Combined_Sentiment_Intent_all_data.csv"
# =============================================================================

sentiment = pd.read_csv(SENTIMENT_CSV)
intent    = pd.read_csv(INTENT_CSV)

# Take post, comment, timestamp and sentiment columns
sent = sentiment[["Post", "Comment", "Comments_time", "Sentiment over comment from models"]]

# Take intent label and per-model probability columns
intent = intent[[
    "final_label_three_models_final",
    "GRO_NLP_prob_Appreciation", "GRO_NLP_prob_Criticism",
    "GRO_NLP_prob_Inquiry",      "GRO_NLP_prob_Statement",
    "roberta_prob_Appreciation",  "roberta_prob_Criticism",
    "roberta_prob_Inquiry",       "roberta_prob_Statement",
    "debertaV3_prob_Appreciation","debertaV3_prob_Criticism",
    "debertaV3_prob_Inquiry",     "debertaV3_prob_Statement",
]]

# Combine the two DataFrames side-by-side and save
combined_df = pd.concat([sent, intent], axis=1)
combined_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved combined data -> {OUTPUT_CSV}")
