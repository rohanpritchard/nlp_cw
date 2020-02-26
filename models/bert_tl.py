from simpletransformers.classification import ClassificationModel
import pandas as pd

from utils.tools import get_data
from scipy.stats.stats import pearsonr


def getTheData(name):
    data = get_data(name, german=False)
    return pd.DataFrame(data, columns=['text_a', 'text_b', 'labels']), [s for e, c, s in data]


train_df, s = getTheData("train")

eval_df, s_val = getTheData("dev")

train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 3,

    'regression': True,
}

# Create a ClassificationModel
model = ClassificationModel('roberta', 'roberta-base', num_labels=1, use_cuda=False, args=train_args)
print(train_df.head())

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print("PEARSON:", pearsonr(s_val, model_outputs))
# predictions, raw_outputs = model.predict([["I'd like to puts some CD-ROMS on my iPad, is that possible?'", "Yes, but wouldn't that block the screen?"]])
# print(predictions)
# print(raw_outputs)