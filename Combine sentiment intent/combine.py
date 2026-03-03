import pandas as pd

sentiment = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\sentiment pred\Sentiments_cleaned_comments.csv")
intent = pd.read_csv(r"C:\Users\20245179\OneDrive - TU Eindhoven\Research Paper\intent pred\intent_predictions_fine_tuned_4_intents_real_data_distribution_majority_voting_3_models.csv")


print(intent.columns)
# take comments posts and sentiment over comment from models columns and final_labels_three_model_final column from inten
sent = sentiment[["Post", "Comment", "Comments_time", "Sentiment over comment from models"]]
intent = intent[[ "final_label_three_models_final", "GRO_NLP_prob_Appreciation","GRO_NLP_prob_Criticism","GRO_NLP_prob_Inquiry","GRO_NLP_prob_Statement", "roberta_prob_Appreciation","roberta_prob_Criticism","roberta_prob_Inquiry","roberta_prob_Statement", "debertaV3_prob_Appreciation","debertaV3_prob_Criticism","debertaV3_prob_Inquiry","debertaV3_prob_Statement"]]
# Combine the two DataFrames
combined_df = pd.concat([sent, intent], axis=1)


# Save the combined DataFrame to a new CSV file
combined_df.to_csv("Combined_Sentiment_Intent_all_data.csv", index=False)