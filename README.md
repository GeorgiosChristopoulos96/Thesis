# MSc Thesis: Data-to-text generation for under-resourced languages

This project is part of the MSc thesis "Data-to-text generation for under-resourced languages" 
which aims to explore many different finetuning scenarios for the mT5 model on the WebNLG dataset. The main focus is if the language families can enhance the D2T performance of the mT5 model.



## Cloning the repository
The following command will clone the repository to your local machine.
```bash
git clone https://github.com/GeorgiosChristopoulos96/Thesis.git
```
## Installing the requirements
The following command will install all the necessary requirements for the project.
```bash
cd Code
pip3 install -r requirements.txt
```
## Before execution
We provide the code of the thesis project as python files which are not parametrized. In order to run the code, you need to change the paths in the code to match your local paths. 
At the begging of each python file, you can find the ```PATH``` variable which you need to change.
### WebNLG dataset
The WebNLG dataset is not included in the repository. You can download it from the following link: [WebNLG](https://github.com/WebNLG/2023-Challenge.git). After downloading the dataset, you should place it in the `Data` directory.
In the `Datasets`  you can find all the datasets used for the finetuning recipes of our experiments.\
*`English_WebNLG`: Is the English only WebNLG dataset.\
*`English_Russian_WebNLG`: Is the English and Russian WebNLG datasets combined into one.\
*`Augmented_WebNLG`: Is WebNLG dataset which was augmented according to the OPUS-100 languages + German using the [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M) model from Meta.
*`Extra_finetuning`: Is the dataset used for the extra finetuning of the mT5 model.\
*`Evaluation_dataset`: Is the dataset used for the evaluation of the mT5 model.


### OPUS-100 dataset
The OPUS-100 dataset is not included in the repository. In the `scripts` directory of our project, you can find the `download_opus.py` script which downloads the dataset. You can run the following command to download  and filter the dataset with the WebNLG related languages.\
```python
cd scirpts
python3 download_opus.py
```
After downloading the dataset, you should place it in the `Data` directory.

### Cleaning the OPUS-100 dataset
The OPUS-100 dataset contains many different language pairs. In the `scripts` directory of our project, you can find the `data_json_cleaning_same_trans.py` script which cleans the dataset from faulty translations. You can run the following command to filter the dataset.
```python
python3 data_json_cleaning_same_trans.py
```
## Code execution

### Pre-training
The following command will execute the code for the pretraining of the mT5 model on the OPUS-100 dataset.
```python
cd Code
python3 mt5_pretraining.py
```

### Finetuning
The following command will execute the code for the finetuning of the mT5 model on the WebNLG dataset.
```python
cd Code
python3 mt5_finetuning.py
```
### Inference
The following command will execute the code for the evaluation of the mT5 model on the WebNLG test dataset.
```python
cd Code
python3 Inference.py
```

## Results
The results of the experiments can be found in the `Model_Outputs` directory of our project. The results are the generated files for each experiment.
For more details on the experiments read the paper.

## Model Checkpoints
The model checkpoints can be found in the following repository from HuggingFace: [Model Checkpoints](https://huggingface.co/GeorgiosChris/Thesis/tree/main)
```bash
git lfs install
```
```bash
git clone https://huggingface.co/GeorgiosChris/Thesis
```

## PARENT Metric Evaluation
The PARENT metric evaluation can be found in the following repository: [PARENT Metric](https://github.com/google-research/language/tree/master/language/table_text_eval)
The files from the repository are required for the evaluation of the PARENT metric. The files should be placed in a new directory named `PARENT_metric`.


