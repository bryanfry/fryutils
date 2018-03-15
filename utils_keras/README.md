# TraceableModel & TraceableResult

I wrote this code a couple weeks ago with the aim to improve the 'traceability' of Keras models trained on an AWS instance for a Kaggle problem.  (Tensorboard might offer better approach to this problem... didn't look there until this was mostly done!)

The code in `traceable_model.py` mainly comprises two classes, as follows:

### `TraceableModel` inherits from the Keras `Sequential` base class.  

The class adds a new `train()` method that handles the following: 

* one-hot encoding for categorical labels / features
* stratified train-test split
* the native `Sequential.fit()` method from Keras. 
* It also adds a Keras `Callback` to remember the training / testing time for each epoch.  

`train()` returns a `TraceableResult` object, described below:

### `TraceableResult` is a container for training accuracy, loss, and time, as well as the training and eval data.
	
This class support methods to do two tasks:

* Pickles itself, so the file can be transferred from remote instance, cluster, etc. for local analysis.  The `TraceableResult` pickle is saved along with `.json` and `.h5` files for the model topology and weights.
* Makes a quick matplotlib plot showing accuracy, loss, and timing info.

### `traceable_model_iris_example.ipynb` 
This is a short Jupyter notebook showing the application of these classes to the Iris classification problem.  It runs in a few seconds locally (very simple model, tiny dataset).