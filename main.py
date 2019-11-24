from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
import argparse
from sklearn import svm
from utils import pre_process_data

def split_train_test_for_main_side_model(X_train, y_train):
    outlier_y_train = []
    outlier_x_train = []

    outliers = ", ".join(['__label__tai_chinh', '__label__kinh_doanh_va_cong_nghiep'])

    for label, content in zip(y_train, X_train):
        if outliers.count(label) == 1:
            outlier_y_train.append(label)
            outlier_x_train.append(content)
    temp_y_train = ['__label__outlier' if outliers.count(label) else label
                for label in y_train]

    return outlier_y_train, outlier_x_train, temp_y_train

def train_main_model(X_train, temp_y_train):
    Tfidf_vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    Tfidf_vect.fit(X_train)
    Train_X_Tfidf = Tfidf_vect.transform(X_train)

    SVM = svm.SVC(C=1, gamma=1, kernel='linear')
    SVM.fit(Train_X_Tfidf, temp_y_train)

    return SVM, Tfidf_vect

def train_side_model(outlier_x_train, outlier_y_train):
    Tfidf_vect_outlier = TfidfVectorizer(max_features=1000, ngram_range=(1,2), max_df=0.5, sublinear_tf=True)
    Tfidf_vect_outlier.fit(outlier_x_train)
    Train_X_outlier_Tfidf = Tfidf_vect_outlier.transform(outlier_x_train)
    
    SVM_outlier = svm.SVC(C=1, gamma=1.8, kernel='sigmoid')
    SVM_outlier.fit(Train_X_outlier_Tfidf, outlier_y_train)
    
    return SVM_outlier, Tfidf_vect_outlier

parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str,
 help="File that you use to train the model")
parser.add_argument('--output_main', type=str,
 help="Name of the file that you save the main training model")
parser.add_argument('--output_side', type=str,
 help="Name of the file that you save the side training model")
parser.add_argument('--stopwords', type=str,
 help="File cotains list of stopwords")

FLAGS = parser.parse_args()

if __name__ == "__main__":
    data = pre_process_data(FLAGS.input, FLAGS.stopwords)
    
    outlier_y_train, outlier_x_train, temp_y_train = split_train_test_for_main_side_model(data['content'], data['label'])

    SVM, Tfidf_vect = train_main_model(data['content'], temp_y_train)
    SVM_outlier, Tfidf_vect_outlier = train_side_model(outlier_x_train, outlier_y_train)

    dump(SVM, FLAGS.output_main + '.joblib')
    dump(SVM_outlier, FLAGS.output_side + '.joblib')
    dump(Tfidf_vect, 'main_tfidf.joblib')
    dump(Tfidf_vect_outlier, 'side_tfidf.joblib')
    print('Finished!')
    