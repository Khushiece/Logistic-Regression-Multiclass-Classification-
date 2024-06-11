<h1>Introduction</h1>
    <p>This project demonstrates the use of logistic regression for multiclass classification using the digits dataset from the sklearn library. The digits dataset contains images of handwritten digits (0-9) represented as 8x8 pixel grids. Each image is transformed into a 64-dimensional vector for classification.</p>
    <br>
    <h1>Dataset Overview</h1>
    <p>We use the <code>load_digits</code> function from sklearn to load the dataset. The dataset consists of:</p>
    <ul>
        <li><strong>Images:</strong> 8x8 arrays of grayscale pixel values representing the handwritten digits.</li>
        <li><strong>Data:</strong> Flattened 64-dimensional vectors corresponding to each image.</li>
        <li><strong>Target:</strong> The digit labels (0-9) for each image.</li>
        <li><strong>Target Names:</strong> The labels for the target values (0-9).</li>
    </ul>
    <p>To visualize the dataset, we use <code>matplotlib</code> to display the first few images.</p>
    <br>
    <h1>Data Exploration</h1>
    <p>We inspect the dataset's attributes to understand its structure. The <code>digits.data</code> contains the flattened pixel values, while <code>digits.target</code> holds the actual digit labels. Here’s a look at the first data point's pixel values and its corresponding target value:</p>
    <pre>
        <code>
            print(digits.data[0])
            print(digits.target[0])
        </code>
    </pre>
    <br>
    <h1>Model Training</h1>
    <h2>Splitting the Data</h2>
    <p>We split the dataset into training and testing sets using <code>train_test_split</code>, reserving 20% of the data for testing:</p>
    <pre>
        <code>
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
        </code>
    </pre>
    <br>
    <h2>Training the Model</h2>
    <p>We create and fit a logistic regression model to the training data:</p>
    <pre>
        <code>
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
        </code>
    </pre>
    <br>
    <h2>Evaluating the Model</h2>
    <p>To evaluate the model, we measure its accuracy on the test data:</p>
    <pre>
        <code>
            accuracy = model.score(X_test, y_test)
            print(f"Model Accuracy: {accuracy}")
        </code>
    </pre>
    <p>The model achieves an accuracy of approximately 93.3%.</p>
    <br>
    <h1>Prediction</h1>
    <p>We predict the digit labels for the first few samples in the test set and compare them with the true labels:</p>
    <pre>
        <code>
            predictions = model.predict(X_test[:5])
            print("Predicted labels: ", predictions)
            print("True labels: ", y_test[:5])
        </code>
    </pre>
    <br>
    <h1>Performance Metrics</h1>
    <p>To gain deeper insights into the model's performance, we use a confusion matrix. This matrix compares the actual versus predicted labels, showing the model's accuracy for each digit:</p>
    <pre>
        <code>
            from sklearn.metrics import confusion_matrix
            import seaborn as sn
            y_predicted = model.predict(X_test)
            cm = confusion_matrix(y_test, y_predicted)
            plt.figure(figsize=(10,7))
            sn.heatmap(cm, annot=True, cmap='coolwarm', fmt='d')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
        </code>
    </pre>
    <br>
    <h1>Conclusion</h1>
    <p>We successfully trained a logistic regression model to classify handwritten digits with high accuracy. The confusion matrix shows that while the model generally predicts the correct digit, it occasionally confuses similar-looking digits.</p>
    <br>
    <h1>Possible Extensions</h1>
    <p>To improve the model, consider the following:</p>
    <ul>
        <li>Experimenting with other classification models like Random Forest or Neural Networks.</li>
        <li>Tuning hyperparameters for better performance.</li>
        <li>Applying more complex image preprocessing techniques.</li>
    </ul>
    <br>
    <h1>Prerequisites</h1>
    <ul>
        <li>Python (>=3.6)</li>
        <li>Scikit-learn (>=0.24)</li>
        <li>Matplotlib (>=3.3)</li>
        <li>Seaborn (>=0.11)</li>
    </ul>
    <br>
    <h1>Installation</h1>
    <ol>
        <li>Install Python from <a href="https://www.python.org/">python.org</a>.</li>
        <li>Install the required libraries using pip:</li>
        <pre>
            <code>
                pip install scikit-learn matplotlib seaborn
            </code>
        </pre>
    </ol>
    <br>
    <h1>Usage</h1>
    <ol>
        <li>Clone this repository to your local machine.</li>
        <li>Navigate to the directory containing the project files.</li>
        <li>Run the Jupyter notebook or Python script to see the results:</li>
        <pre>
            <code>
                jupyter notebook digits_classification.ipynb
                # or
                python digits_classification.py
            </code>
        </pre>
    </ol>
    <br>
    <h1>References</h1>
    <ul>
        <li>Scikit-learn Documentation: <a href="https://scikit-learn.org/stable/">Scikit-learn</a></li>
        <li>Matplotlib Documentation: <a href="https://matplotlib.org/stable/contents.html">Matplotlib</a></li>
        <li>Seaborn Documentation: <a href="https://seaborn.pydata.org/">Seaborn</a></li>
    </ul>
