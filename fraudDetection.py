import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

class fraudDetection:
    def __init__(self):
        self.getDataSet()
        self.model = self.design_model()
        print("Model Created with optimum params")
    
    def design_model(self):
        model = RandomForestClassifier()
        print("Fitting Data")
        model.fit(self.features_train, self.labels_train)
        print("Dataset done fitting")
        joblib.dump(model, 'fraudDetectionModel.pkl')
        print("model dumped to fraudDetectionModel.pkl")
        return model
    
    def getDataSet(self):
        dataset = pd.read_csv('creditCardDataCopy.csv') #load the dataset
        features = dataset.iloc[:,0:7] #choose first 8 columns as features
        labels = dataset.iloc[:,-1] #choose the final column for prediction
        #one-hot encoding for categorical variables
        features = pd.get_dummies(features) 
        #split data into testing and training
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) #split the data into training and test data

        #standardize if needed (play around to see if it helps accuracy)
        # ct = ColumnTransformer([('standardize', StandardScaler(), ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])], remainder='passthrough')
        # self.features_train = ct.fit_transform(self.features_train)
        # self.features_test = ct.transform(self.features_test)
    
    def runModel(self):
        #do what you want with the model
        testRow = pd.read_csv('creditCardDataCopyTest.csv')
        testRow = testRow.drop("isFraud", axis=1)
        print(testRow)

        #self.model.predict(testRow)

fraudAlgorithm = fraudDetection()
fraudAlgorithm.runModel()
