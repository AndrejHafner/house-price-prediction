import pandas as pd
import numpy as np
import os
import re
import itertools


BASE_SUBMISSION_DIR = "./submissions"
SVM_SUBMISSION_DIR = BASE_SUBMISSION_DIR + "/svm"
LIGHTGBM_SUBMISSION_DIR = BASE_SUBMISSION_DIR + "/lightgbm"
XGBOOST_SUBMISSION_DIR = BASE_SUBMISSION_DIR + "/xgboost"
RANDOM_FOREST_SUBMISSION_DIR = BASE_SUBMISSION_DIR + "/random_forest"

def save_submission(dir, df_test, predictions, testing_results = None):
    if not os.path.exists(dir): os.mkdir(dir)

    submission = pd.DataFrame({"Id": df_test["Id"], "SalePrice": np.array(predictions)})
    directory_contents = os.listdir(dir)
    submission_numbers = [re.findall(r'\d+', el) for el in directory_contents]
    numbers = list(itertools.chain(*submission_numbers))

    latest_submission_number = int(max(numbers)) if len(numbers) > 0 else 0

    submission.to_csv(f"{dir}/{latest_submission_number + 1}.csv", index=False)
    if testing_results is not None:
        with open(f"{dir}/{latest_submission_number + 1}_test_results.txt","w+") as file:
            file.write(testing_results)
            file.close()