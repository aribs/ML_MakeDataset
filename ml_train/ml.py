from models import DlModels
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.preprocessing import sequence
import keras.callbacks as ckbs
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

TEST_RESULTS = {'data': {},
                "embedding": {},
                "hiperparameter": {},
                "test_result": {}}

class Plotter:

    def plot_graphs(self, train, val, save_to=None, name="accuracy"):

        if name == "accuracy":
            val, = plt.plot(val, label="val_acc")
            train, = plt.plot(train, label="train_acc")
        else:
            val, = plt.plot(val, label="val_loss")
            train, = plt.plot(train, label="train_loss")

        plt.ylabel(name)
        plt.xlabel("epoch")

        plt.legend(handles=[val, train], loc=2)

        if save_to:
            plt.savefig("{0}/{1}.png".format(save_to, name))

        plt.close()

    def plot_confusion_matrix(self, confusion_matrix, categories, save_to=None, normalized=False):

        sns.set()
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(14.0, 7.0))

        if normalized:
            row_sums = np.asanyarray(confusion_matrix).sum(axis=1)
            matrix = confusion_matrix / row_sums[:, np.newaxis]
            matrix = [line.tolist() for line in matrix]
            g = sns.heatmap(matrix, annot=True, fmt='f', xticklabels=True, yticklabels=True)

        else:
            matrix = confusion_matrix
            g = sns.heatmap(matrix, annot=True, fmt='d', xticklabels=True, yticklabels=True)

        g.set_yticklabels(categories, rotation=0)
        g.set_xticklabels(categories, rotation=90)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        if save_to:
            if normalized:
                plt.savefig("{0}/{1}.png".format(save_to, "normalized_confusion_matrix"))
            else:
                plt.savefig("{0}/{1}.png".format(save_to, "confusion_matrix"))

class CustomCallBack(ckbs.Callback):

    def __init__(self):
        ckbs.Callback.__init__(self)
        TEST_RESULTS['epoch_times'] = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        TEST_RESULTS['epoch_times'].append(time.time() - self.epoch_time_start)


class PhisingDetection:
    def __init__(self):

        self.params = {
            'loss_function': 'binary_crossentropy',
            'sequence_length': 18,
            'batch_train': 5000,
            'batch_test': 5000,
            'categories': ['phishing'],
            'char_index': None,
            'epoch': 30,
            'embedding_dimension': 50,
        }
        self.ml_plotter = Plotter()
        self.dl_models = DlModels(self.params['categories'], self.params['embedding_dimension'], self.params['sequence_length'])

    def load_and_prepare_data(self, path_to_data):
        # Cargar el conjunto de datos
        df = pd.read_csv(path_to_data)

        # Convertir las columnas de fecha de string a datetime y extraer características
        df['creation_date'] = pd.to_datetime(df['creation_date'], format='%Y/%m/%d')
        df['updated_date'] = pd.to_datetime(df['updated_date'], format='%Y/%m/%d')

        # Extraer características de las fechas
        df['creation_year'] = df['creation_date'].dt.year
        df['creation_month'] = df['creation_date'].dt.month
        df['creation_day'] = df['creation_date'].dt.day
        df['updated_year'] = df['updated_date'].dt.year
        df['updated_month'] = df['updated_date'].dt.month
        df['updated_day'] = df['updated_date'].dt.day

        # Opcional: Diferencia en días entre creation_date y updated_date
        df['days_difference'] = (df['updated_date'] - df['creation_date']).dt.days

        df = df.drop(columns=['creation_date', 'updated_date', 'message'])

        X = df

        #Shuffle the data
        X = X.sample(frac=1).reset_index(drop=True)

        # La variable objetivo 'is_smsing' es parte de X, la separamos para tener y
        y = X['is_smsing']

        X = X.drop(['is_smsing'], axis=1)

        return X, y, df

    def randomise_data(self, path):
        df = pd.read_csv(path)
        # Convertir las columnas de fecha de string a datetime y extraer características
        df['creation_date'] = pd.to_datetime(df['creation_date'], format='%Y/%m/%d')
        df['updated_date'] = pd.to_datetime(df['updated_date'], format='%Y/%m/%d')

        # Extraer características de las fechas
        df['creation_year'] = df['creation_date'].dt.year
        df['creation_month'] = df['creation_date'].dt.month
        df['creation_day'] = df['creation_date'].dt.day
        df['updated_year'] = df['updated_date'].dt.year
        df['updated_month'] = df['updated_date'].dt.month
        df['updated_day'] = df['updated_date'].dt.day
        # Opcional: Diferencia en días entre creation_date y updated_date
        df['days_difference'] = (df['updated_date'] - df['creation_date']).dt.days

        #Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv('random_val.csv', index=True) 

    def set_parameters(self, args):
        self.params['epoch'] = int(args.epoch)
        self.params['architecture'] = args.architecture
        self.params['batch_train'] = args.batch_size
        self.params['batch_test'] = args.batch_size

    def model_sum(self, x):
        try:
            TEST_RESULTS['hiperparameter']["model_summary"] += x
        except:
            TEST_RESULTS['hiperparameter']["model_summary"] = x

    def algorithm(self, x_train, y_train, x_val, y_val, x_test, y_test):

        # x_train = sequence.pad_sequences(x_train, maxlen=self.params['sequence_length'])
        # x_test = sequence.pad_sequences(x_test, maxlen=self.params['sequence_length'])
        # x_val = sequence.pad_sequences(x_val, maxlen=self.params['sequence_length'])

        print("train sequences: {}  |  test sequences: {} | val sequences: {}\n"
              "x_train shape: {}  |  x_test shape: {} | x_val shape: {}\n"
              "Building Model....".format(len(x_train), len(x_test), len(x_val), x_train.shape, x_test.shape, x_val.shape))

        model = eval("self.dl_models.{}(self.params['char_index'])".format(self.params['architecture']))

        model.compile(loss=self.params['loss_function'], optimizer='adam', metrics=['accuracy'])

        model.summary()
        model.summary(print_fn=lambda x: self.model_sum(x + '\n'))
    

        x_numpy = x_train.to_numpy()
        y_numpy = y_train.to_numpy()
        x_numpy = np.delete(x_numpy, 0, 0).astype(float)
        y_numpy = np.delete(y_numpy, 0, 0).astype(float)

        
        hist = model.fit(x_numpy, y_numpy,
                         batch_size=self.params['batch_train'],
                         epochs=self.params['epoch'],
                         shuffle=True,
                         validation_data=(x_val, y_val),
                         callbacks=[CustomCallBack()])

        t = time.time()
        score, acc = model.evaluate(x_test, y_test, batch_size=self.params['batch_test'])

        TEST_RESULTS['test_result']['test_time'] = time.time() - t

        y_test = list(np.argmax(np.asanyarray(np.squeeze(y_test), dtype=int).tolist(), axis=1))
        y_pred = model.predict_classes(x_test, batch_size=self.params['batch_test'], verbose=1).tolist()
        report = classification_report(y_test, y_pred, target_names=self.params['categories'])
        print(report)
        TEST_RESULTS['test_result']['report'] = report
        TEST_RESULTS['epoch_history'] = hist.history
        TEST_RESULTS['test_result']['test_acc'] = acc
        TEST_RESULTS['test_result']['test_loss'] = score

        test_confusion_matrix = confusion_matrix(y_test, y_pred)
        TEST_RESULTS['test_result']['test_confusion_matrix'] = test_confusion_matrix.tolist()

        print('Test loss: {0}  |  test accuracy: {1}'.format(score, acc))
        self.save_results(model)

    def load_and_vectorize_data(self):
        print("data loading")
        train = pd.read_csv('random_train.csv', header=None)
        test = pd.read_csv('random_predict.csv',header=None)
        val = pd.read_csv('random_val.csv', header=None)

        TEST_RESULTS['data']['samples_train'] = len(train)
        TEST_RESULTS['data']['samples_test'] = len(test)
        TEST_RESULTS['data']['samples_val'] = len(val)
        TEST_RESULTS['data']['samples_overall'] = len(train) + len(test) + len(val)
        TEST_RESULTS['data']['name'] = '/'

        #Select all columns but first
        raw_x_train = train.iloc[:, 1:]
        #Select first column
        raw_y_train = train.iloc[:, 0]

        #Select all columns but first
        raw_x_val = val.iloc[:, 1:]
        #Select first column
        raw_y_val = val.iloc[:, 0]

        #Select all columns but first
        raw_x_test = test.iloc[:, 1:]
        #Select first column
        raw_y_test = test.iloc[:, 0]


        x_train = raw_x_train
        x_val = raw_x_val
        x_test = raw_x_test

        encoder = LabelEncoder()
        encoder.fit(self.params['categories'])
        y_train = raw_y_train
        y_val = raw_y_val
        y_test = raw_y_test
        print("Data are loaded.")

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--epoch", default=10, help='The number of epoch')
    parser.add_argument("-arch", "--architecture", help='Architecture function in models.py', required=True)
    parser.add_argument("-bs", "--batch_size", default=1000, help='batch size', type=int)

    args = parser.parse_args()

    return args


def main():
    args = argument_parsing()
    phising_detector = PhisingDetection()
    phising_detector.set_parameters(args)
    # phising_detector.randomise_data('data_to_test.csv')
    #Preparing train and validation data
    X, y, df = phising_detector.load_and_prepare_data('dataset.csv')
    X_val, y_val, df_val = phising_detector.load_and_prepare_data('data_to_test.csv')

    (x_train_2, y_train_2), (x_val_2, y_val_2), (x_test_2, y_test_2) = phising_detector.load_and_vectorize_data()
    #Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # phising_detector.algorithm(x_train_2, y_train_2, x_val_2, y_val_2, x_test_2, y_test_2)

    #Chose model
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model_2 = HistGradientBoostingClassifier()

    #Fit the model
    model.fit(X_train, y_train)
    model_2.fit(X_train, y_train)

    #Predict
    y_pred = model.predict(X_test)
    y_pred_2 = model_2.predict(X_test)

    #Check accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_2 = accuracy_score(y_test, y_pred_2)
    print(f'Accuracy Model 1: {accuracy}')
    print(f'Accuracy Model 2: {accuracy_2}')

    #Predict using the pre-trained model
    predicted_classes = model.predict(X_val)
    predicted_classes_2 = model_2.predict(X_val)

    df_val['Predicted_is_smsing'] = predicted_classes

    # Create Csv file
    df_val.to_csv('predicted_decission_tree.csv', index=True)      
    print("Prediction saved in predicted_decission_tree.csv")

    df_val['Predicted_is_smsing'] = predicted_classes_2
    df_val.to_csv('predicted_gradient_boosting.csv', index=True)
    print("Prediction saved in predicted_gradient_boosting.csv")


if __name__ == '__main__':
    main()

# Ex to execute the cnn arquitecture 
# python3 ml.py -arch cnn_base -ep 30 -bs 1000