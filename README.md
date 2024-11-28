# 🟡 Classification of The Simpsons Characters Using a CNN

---

## **🌟Description**
- **Main purpose**: Image Classification.
- **Custom CNN Architecture**: VGG11BN, ResNet.
- **Completed dataset**: Manually added pictures and made augmentation.

---

## 📂 Project Structure
```plaintext
├── config.yaml               # Configuration file for the model and training
├── dataset_transform.py      # Data processing script
├── graphs.py                 # Graph plotting utilities
├── imports.py                # Common imports and variables
├── main.py                   # Main script to run the pipeline
├── model.py                  # Model architecture definition
├── preparing_model.py        # Model preparation and object initialization
├── README.md                 # Project description
├── requirements.txt          # Project dependencies file
├── testing.py                # Script for model testing
├── testing_jupiter.ipynb     # Jupyter Notebook for interactive testing
├── train_and_test.sh         # Shell script for training and testing
├── training.py               # Model training script
├── utilities.py              # Utility functions
└── yaml_reader.py            # YAML configuration file parser
```

---

## 🛠 Installation
### 1. Clone the Repository:
`git clone https://github.com/pxleblank/CNN_for_detection_the_simpsons_characters.git`
### 2. Set Up a Virtual Environment:
`python -m venv venv`
`source venv/bin/activate`. On Windows, use `venv\\Scripts\\activate`
### 3. Install Dependencies:
`pip install -r requirements.txt`
### 4. Download the Dataset:
[https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset?resource=download](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset?resource=download)

---

## 🚀 Usage
### 1. Configure the Project:
Edit the _config.yaml_ file to set parameters such as learning rate, batch size, model hyperparameters e.t.c. You can also change something in other files.
### 2. Run training and testing:
Run _train_and_test.sh_ or _main.py_.
### 3. Look at the graphs:
If necessary, you can save the graphs.

---

## 📊 Results and Visualizations
### All graphs are shown for the VGG11BN model.

![](https://github.com/pxleblank/CNN_for_detection_the_simpsons_characters/blob/master/graphs/trainvalloss.png?raw=true)
![](https://github.com/pxleblank/CNN_for_detection_the_simpsons_characters/blob/master/graphs/trainvalacc.png?raw=true)
![](https://github.com/pxleblank/CNN_for_detection_the_simpsons_characters/blob/master/graphs/trainvalprecisionrecall.png?raw=true)
![](https://github.com/pxleblank/CNN_for_detection_the_simpsons_characters/blob/master/graphs/Precision_for_each_class_(Last_Epoch).png?raw=true)
![](https://github.com/pxleblank/CNN_for_detection_the_simpsons_characters/blob/master/graphs/Recall_for_each_class_(Last_Epoch).png?raw=true)