from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import *
import numpy as np
import os
from matplotlib import pyplot

# Evolutation part
def evolutation_function(model_path,x_train,y_train):
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    predicted_y = model_path.predict(x_train)
    predicted_y = np.array([float(round(i[0])) for i in predicted_y])

    accuracy_score_test = round(accuracy_score(y_train, predicted_y),2)
    print("Accuracy Score of the Model : ",str(int(accuracy_score_test*100))+" %")

    print("Complete result given below: \n");
    print('Accuracy:', accuracy_score(y_train, predicted_y))

    print ('F1 score:', f1_score(y_train, predicted_y,
                                average='weighted'))

    print ('Recall:', recall_score(y_train, predicted_y,
                                average='weighted'))

    print ('Precision:', precision_score(y_train, predicted_y,
                                        average='weighted'))

    target_names = ['class of Good Image', 'class of Not Good Image']
    print ('\n clasification report:\n', classification_report(y_train, predicted_y,target_names=target_names))

    # print ('\n confussion matrix:\n',confusion_matrix(y_train, predicted_y))

    print("\nConfusion Matrix: \n");
    cm = confusion_matrix(y_train, predicted_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    pyplot.show()
    
