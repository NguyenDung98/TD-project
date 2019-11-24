import argparse
from joblib import load
from utils import pre_process_test_data, store_file
from sklearn.metrics import classification_report

def get_side_test(X_test):
    outlier_true_x_test = []
    x_list = X_test.tolist()
    for index, label in enumerate(predictions_SVM):
        if label == '__label__outlier':
            outlier_true_x_test.append(x_list[index])
    return outlier_true_x_test

def get_final_result(predictions_SVM, predictions_SVM_outlier):
    predictions_SVM_outlier = predictions_SVM_outlier.tolist()
    predictions = []
    for label in predictions_SVM:
        if label == '__label__outlier':
            predictions.append(predictions_SVM_outlier.pop(0))
        else:
            predictions.append(label)
    return predictions

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str,
 help="File that you use to test the model")
parser.add_argument('--stopwords', type=str,
 help="File cotains list of stopwords")
parser.add_argument('--main', type=str,
 help="Main model that you used to train")
parser.add_argument('--side', type=str,
 help="Side model attached with the main model")
parser.add_argument('--main_vector', type=str,
 help="Vector for main model")
parser.add_argument('--side_vector', type=str,
 help="Vector for side model")

FLAGS = parser.parse_args()

if __name__ == "__main__":
    data = pre_process_test_data(FLAGS.input, FLAGS.stopwords)

    SVM = load(FLAGS.main)
    SVM_outlier = load(FLAGS.side)
    Tfidf_vect = load(FLAGS.main_vector)
    Tfidf_vect_outlier = load(FLAGS.side_vector)

    Test_X_Tfidf = Tfidf_vect.transform(data['content'])
    predictions_SVM = SVM.predict(Test_X_Tfidf)

    outlier_true_x_test = get_side_test(data['content'])
    Test_X_outlier_Tfidf = Tfidf_vect_outlier.transform(outlier_true_x_test)
    predictions_SVM_outlier = SVM_outlier.predict(Test_X_outlier_Tfidf)

    predictions = get_final_result(predictions_SVM, predictions_SVM_outlier)

    store_file('predictions.txt', predictions, [""] * len(predictions))
    print('predictions is saved in "predictions.txt"')
