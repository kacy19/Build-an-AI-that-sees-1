import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def predict_and_save(model_path, test_data, output_path):
    model = load_model(model_path)
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    submission = pd.DataFrame({"ImageId": list(range(1, len(predicted_labels)+1)),
                               "Label": predicted_labels})
    submission.to_csv(output_path, index=False)
