import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from CustomSupportVectorMachine import CustomSVM, CustomSVM_OVR

def run_svm_tuning_example():
    # Load Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

    X = df.drop(['target', 'flower_name'], axis=1).values
    y = df.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grids
    lambda_params = [0.001, 0.01, 0.1, 1.0]
    gammas = [0.01, 0.1, 1.0, 10.0] # Only relevant for RBF kernel
    kernels = ['linear', 'rbf']

    best_score = 0
    best_params = {}

    print("Starting Grid Search for CustomSVM_OVR...")

    for kernel in kernels:
        for lambda_param in lambda_params:
            if kernel == 'rbf':
                for gamma in gammas:
                    print(f"Testing: kernel={kernel}, lambda_param={lambda_param}, gamma={gamma}")
                    svm = CustomSVM_OVR(learning_rate=0.001,
                                        lambda_param=lambda_param, n_iterations=1000,
                                        kernel=kernel, gamma=gamma)
                    svm.fit(X_train, y_train)
                    score = svm.score(X_test, y_test)
                    print(f"Score: {score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_params = {'kernel': kernel,'lambda_param': lambda_param, 'gamma': gamma}
            else: # Linear kernel
                print(f"Testing: kernel={kernel}, lambda_param={lambda_param}")
                svm = CustomSVM_OVR(learning_rate=0.001, 
                                    lambda_param=lambda_param, 
                                    n_iterations=1000,
                                    kernel=kernel)
                svm.fit(X_train, y_train)
                score = svm.score(X_test, y_test)
                print(f"Score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = {'kernel': kernel,'lambda_param': lambda_param}

    print("\nGrid Search Complete.")
    print(f"Best Score: {best_score * 100:.2f}%")
    print(f"Best Parameters: {best_params}")

    # Train the final model with the best parameters
    print("\nTraining final model with best parameters...")
    final_svm = CustomSVM_OVR(learning_rate=0.001,
        n_iterations=1000, **best_params)
    final_svm.fit(X_train, y_train)
    final_accuracy = final_svm.score(X_test, y_test)
    print(f"Final Model Accuracy: {final_accuracy * 100:.2f}%")

if __name__ == "__main__":
    run_svm_tuning_example()
