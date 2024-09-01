import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from functions import *
np.random.seed(42)
# File uploading
file_path = '/Users/lorispalmarin/PycharmProjects/MachLear/data.csv'
df = pd.read_csv(file_path)

##### EDA

print("First rows of the dataset:\n", df.head())
print("\nStatistical summary:\n", df.describe())

# Removing NAs, if any
df = df.dropna()

num_columns = df.drop('y', axis=1).columns  # Removing target var Y

# Boxplot of xN variables
plt.figure(figsize=(20, 15))
for i, column in enumerate(num_columns, 1):
    plt.subplot(4, 3, i)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# Boxplot to find outliers
plt.figure(figsize=(20, 15))
for i, column in enumerate(num_columns, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()


### Outliers handling - IQR method
for column in ['x2', 'x7', 'x8', 'x9']: # Cycle for each column when removing outliers
    df = remove_outliers(df, column)

# Getting final dimension of dataset
print(f'Final dataset shape: {df.shape}')


### Check of correlation
correlation_matrix = df.iloc[:, :10].corr()

# Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix pre-processing')
plt.show()

# Removing x6 and x10 because higly correlated
df = df.drop(['x6', 'x10'], axis=1)

# Double check of correlation
correlation_matrix = df.iloc[:, :8].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix post-processing')
plt.show()


### Splitting in train and test set
X = df.drop('y', axis=1).values
y = df['y'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


### Feature scaling
scaler = StandardScaler()
# Fit on training set, then transformation of both traind and test set IMPORTANTE PER DATA LEAKAGE
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Last check
print("\nStatistical summary of training set after standardisation:\n", pd.DataFrame(X_train).describe())

#set seed for replicability of SVM and LogReg
np.random.seed(42)



######## PERCEPTRON w/CROSS VALIDATION

print(f"Running Perceptron with 3 different max_epochs: (1000, 2000, 5000), using 5-fold cross-validation.")


#Initialising models and cross validation
perceptron_model = Perceptron(max_epochs=1000)
perceptron_model2 = Perceptron(max_epochs=2000)
perceptron_model3 = Perceptron(max_epochs=5000)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Executing cross-validation
scores = cross_val_score(perceptron_model, X_train, y_train, cv=kf)
scores2 = cross_val_score(perceptron_model2, X_train, y_train, cv=kf)
scores3 = cross_val_score(perceptron_model3, X_train, y_train, cv=kf)

# Results
print("Accuracy scores for each fold (1000 epochs):", scores)
print("Mean cross-validation accuracy (1000 epochs):", np.mean(scores))
print("Accuracy scores for each fold (2000 epochs):", scores2)
print("Mean cross-validation accuracy (2000 epochs):", np.mean(scores2))
print("Accuracy scores for each fold (5000 epochs):", scores3)
print("Mean cross-validation accuracy (5000 epochs):", np.mean(scores3))

# Choosing best model basing on performances
best_model = perceptron_model

# Training the best model on whole training set
best_model.fit(X_train, y_train)

# Predicting labels on test set
y_pred = best_model.predict(X_test)

# Computing and printing accuracy on test set
accuracy = np.mean(y_test == y_pred)
loss = np.mean( y_test != y_pred)
print(f'Accuracy on test set: {accuracy:.2f}')
print(f'Misclassification rate on test set: {loss}')






######## SVM w/PEGASOS UPDATES

print(f"\nRunning Support Vector Machine with different values of parameters \nUsing grid search with 5-fold cross validation:")


# Defining different combinations of parameters
param_grid = {'lambda_param': [0.001, 0.01, 0.1, 1],
              'max_iter': [1000, 2000, 5000]}
print(param_grid)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iteration over all combinations of parameters
best_score = -np.inf
best_params = None

for lambda_param in param_grid['lambda_param']:
    for max_iter in param_grid['max_iter']:

        # Initialising model with different parameters
        pegasos_svm = PegasosSVM(lambda_param=lambda_param, max_iter=max_iter)

        # Cross-validation
        scores = cross_val_score(pegasos_svm, X_train, y_train, cv=kf)
        mean_score = np.mean(scores)
        print('Accuracy with lambda (', lambda_param, ') and max_iter (', max_iter, ') is ',mean_score)

        # Updating best scores and parameters
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'lambda_param': lambda_param, 'max_iter': max_iter}


# Printing best Parameters and Accuracy
print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score:.2f}')

# Training model with best parameters
best_model = PegasosSVM(**best_params)
best_model.fit(X_train, y_train)

# Predicting labels of test set
y_pred = best_model.predict(X_test)

# Computing and printing accuracy on test set
accuracy = np.mean(y_test == y_pred)
loss = np.mean( y_test != y_pred)
print(f'Accuracy on test set: {accuracy:.2f}')
print(f'Misclassification rate on test set: {loss}')






######## LOGISTIC REGRESSION w/PEGASOS UPDATES (LOGISTIC LOSS)

print('\nRunning Regularised Logistic Regression with different values of parameters \nGrid search with 5-fold cross validation:')

# Different combinations of parameters to test
param_grid = {'lambda_param': [0.001, 0.01, 0.1, 1],
              'max_iter': [1000, 2000, 5000]}

print(param_grid)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterating over all combinations of parameters
best_score = -np.inf
best_params = None

for lambda_param in param_grid['lambda_param']:
    for max_iter in param_grid['max_iter']:

        # Initialising model with different parameters
        logistic_model = PegasosLogisticRegression(lambda_param=lambda_param, max_iter=max_iter)

        # Cross-validation
        scores = cross_val_score(logistic_model, X_train, y_train, cv=kf)
        mean_score = np.mean(scores)
        print('Accuracy with lambda (', lambda_param, ') and max_iter (', max_iter, ') is ', mean_score)

        # Updating best score and parameters, if necessary
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'lambda_param': lambda_param, 'max_iter': max_iter}


# Printing best Parameters and Accuracy.
print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score:.2f}')

# Training model with best parameters
best_model = PegasosLogisticRegression(**best_params)
best_model.fit(X_train, y_train)

# Predicting labels of test set
y_pred = best_model.predict(X_test)

# Computing and printing accuracy on test set
accuracy = np.mean(y_test == y_pred)
loss = np.mean( y_test != y_pred)
print(f'Accuracy on test set: {accuracy:.2f}')
print(f'Misclassification rate on test set: {loss}')



#### POLYNOMIAL FEATURE EXPANSION OF DEGREE 2

print('\n\nApplying polynomial feature expansion of degree 2')

# Creating new test and training set
X_train_poly = polynomial_feature_expansion(X_train)
X_test_poly = polynomial_feature_expansion(X_test)

print("\nShape of original features:", X_train.shape)
print("Shape of features after polynomial expansion:", X_train_poly.shape)


# Generating features names
feature_names = ['x' + str(i + 1) for i in range(X_train.shape[1])]
poly_feature_names = feature_names[:]

for i in range(len(feature_names)):
    for j in range(i, len(feature_names)):
        poly_feature_names.append(feature_names[i] + '*' + feature_names[j])

print(f'New list of features: {poly_feature_names}')


### Initializing models and parameters

# Perceptron
perceptron_model = Perceptron(max_epochs=1000)
perceptron_max_iter_grid = [1000, 2000, 5000]

# SVM with Pegasos
svm_model = PegasosSVM(lambda_param=0.01, max_iter=1000)
svm_params = {'lambda_param': [0.001, 0.01, 0.1], 'max_iter': [1000, 2000, 5000]}

# Regularized Logistic Regression with Pegasos
logistic_model = PegasosLogisticRegression(lambda_param=0.01, max_iter=1000)
logistic_params = {'lambda_param': [0.001, 0.01, 0.1], 'max_iter': [1000, 2000, 5000]}

### Hyperparameter tuning for each model

# Perceptron
print('\nRunning Perceptron')
best_params_perceptron = cross_val_model_perceptron(perceptron_model, perceptron_max_iter_grid, X_train, y_train, X_train_poly)
# Retrain model with best parameters
perceptron_model.set_params(**best_params_perceptron)
perceptron_model.fit(X_train_poly, y_train)

# SVM with Pegasos
print('\nRunning SVM')
best_params_svm = cross_val_model(svm_model, svm_params, X_train, y_train, X_train_poly)
#Retrain model with best parameters
svm_model.set_params(**best_params_svm)
svm_model.fit(X_train_poly, y_train)

# Regularized Logistic Regression with Pegasos
print('\nRunning Logistic Regression')
best_params_logistic = cross_val_model(logistic_model, logistic_params, X_train, y_train, X_train_poly)
# Retrain model with best parameters
logistic_model.set_params(**best_params_logistic)
logistic_model.fit(X_train_poly, y_train)

### Comparing weights before and after polynomial expansion

print('\n\nCompare weights before and after polynomial expansion on: \nPERCEPTRON')
compare_weights(perceptron_model, feature_names, poly_feature_names, X_train, X_train_poly, y_train)
print('\nSVM WITH PEGASOS UPDATES')
compare_weights(svm_model, feature_names, poly_feature_names, X_train, X_train_poly, y_train)
print('\nREGULARISED LOGISTIC REGRESSION')
compare_weights(logistic_model, feature_names, poly_feature_names, X_train, X_train_poly, y_train)

### Final evaluation of models on test set

print("\nEvaluation of Perceptron on test set:")
evaluate_model_on_test(perceptron_model, X_train_poly, y_train, X_test_poly, y_test)

print("\nEvaluation of SVM with Pegasos on test set:")
evaluate_model_on_test(svm_model, X_train_poly, y_train, X_test_poly, y_test)

print("\nEvaluation of Logistic Regression on test set:")
evaluate_model_on_test(logistic_model, X_train_poly, y_train, X_test_poly, y_test)




#### KERNELISED VERSION OF PERCEPTRON

# INITIALISING MODELS with polynomial and gaussian kernels
kperceptron_gaussian = KernelPerceptron(max_epochs=1000, kernel=gaussian_kernel)
param_grid = {'gamma': [0.01, 0.1, 1], 'max_epochs': [1000]}
kperceptron_polynomial = KernelPerceptron(max_epochs=1000, kernel=polynomial_kernel)
param_grid_2 = {'degree': [2, 3, 4], 'max_epochs': [1000]}


print('Running Kernelised Perceptron with gaussian kernel: ')
# CV and grid-search on Gaussian model
best_params_kpg = cross_val_model_kernel(kperceptron_gaussian, param_grid, X_train, y_train, kernel_type='gaussian')
kperceptron_gaussian.set_params(**best_params_kpg)
kperceptron_gaussian.fit(X_train, y_train)

print('Running Kernelised Perceptron with polynomial kernel: ')
# CV and grid-search on Polynomial model
best_params_kpp = cross_val_model_kernel(kperceptron_polynomial, param_grid_2, X_train, y_train, kernel_type='polynomial')
kperceptron_polynomial.set_params(**best_params_kpp)
kperceptron_polynomial.fit(X_train, y_train)


# Evaluation on test set

# Predicting labels of test set with POLYONMIAL
y_pred = kperceptron_polynomial.predict(X_test)
# Computing and printing accuracy
accuracy = np.mean(y_test == y_pred)
loss = np.mean(y_test != y_predy)
print(f'Accuracy on test set (polynomial kernel): {accuracy:.2f}')
print(f'Loss on test set (polynomial kernel): {loss:.2f}')


# Predicting labels of test set with GAUSSIAN
y_pred = kperceptron_gaussian.predict(X_test)
# Computing and printing accuracy
accuracy = np.mean(y_test == y_pred)
loss = np.mean(y_test != y_predy)
print(f'Accuracy on test set (gaussian kernel): {accuracy:.2f}')
print(f'Loss on test set (gaussian kernel): {loss:.2f}')



#### KERNELISED VERSION OF PEGASOS FOR SVM

# Definizione del Kernel Pegasos SVM con kernel gaussiano e polinomiale
kpegasos_gaussian = KernelPegasosSVM(max_iter=1000, kernel=gaussian_kernel)
param_grid_gaussian = {'gamma': [0.01, 0.1, 1], 'lambda_param': [0.001, 0.01, 0.1]}

kpegasos_polynomial = KernelPegasosSVM(max_iter=1000, kernel=polynomial_kernel)
param_grid_polynomial = {'degree': [2, 3, 4], 'lambda_param': [0.001, 0.01, 0.1]}

# Funzione per cross-validation e grid search per il kernel gaussiano
print('Running Kernelised Pegasos SVM with gaussian kernel: ')
best_params_kpg = cross_val_model_kernelSVM(kpegasos_gaussian, param_grid_gaussian, X_train, y_train, kernel_type='gaussian')
kpegasos_gaussian.set_params(**best_params_kpg)
kpegasos_gaussian.fit(X_train, y_train)

# Funzione per cross-validation e grid search per il kernel polinomiale
print('Running Kernelised Pegasos SVM with polynomial kernel: ')
best_params_kpp = cross_val_model_kernelSVM(kpegasos_polynomial, param_grid_polynomial, X_train, y_train, kernel_type='polynomial')
kpegasos_polynomial.set_params(**best_params_kpp)
kpegasos_polynomial.fit(X_train, y_train)


# Evaluation on test set

# Predicting labels of test set with POLYOMIAL
y_pred = kpegasos_polynomial.predict(X_test)
# Computing and printing accuracy
accuracy = np.mean(y_test == y_pred)
loss = np.mean(y_test != y_pred)
print(f'Accuracy on test set (polynomial kernel): {accuracy:.2f}')
print(f'Loss on test set (polynomial kernel): {loss:.2f}')



# Predicting labels of test set with GAUSSIAN n
y_pred = kpegasos_gaussian.predict(X_test)
# Computing and printing accuracy
accuracy = np.mean(y_test == y_pred)
loss = np.mean(y_test != y_pred)
print(f'Accuracy on test set (polynomial kernel): {accuracy:.2f}')
print(f'Loss on test set (polynomial kernel): {loss:.2f}')