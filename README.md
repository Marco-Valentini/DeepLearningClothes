# DeepLearningClothes
This repository contains a BERT-like model that can accomplish the fill-in-the-blank task in the fashion domain.
The model will take a sequence of four items (shoes, top, accessory, bottom) with one missing (masked) and predict the missing item given the context.

## Directory tree
* **checkpoints**: In the "checkpoints" folder are saved all the checkpoints of the training of the various models used in the project. In particular, there are the checkpoints of the fine-tuned CNN (Resnet18), the trained AutoEncoder (using the SSIM Loss) and the trained umBERT and WumBERT.
* **dataset**: The "dataset" folder contains all the files and the images of the Polyvore outfits dataset downloaded from the following link: https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/.
* **models**: In the "models" folder there are all the Python classes which define the structure, the methods and the properties of the various models used in the project. In particular, there is the Python class of the CNN (Resnet18) which introduces a few modifications to the original architecture of Resnet18 to obtain a custom dimension of the embeddings, the Python class of the AutoEncoder and the Python classes of umBERT and WumBERT.
* **scripts**: The "scripts" folder contains the Python scripts that must be executed in order to prepare the data and to obtain the embeddings to feed in input to the umBERT (WumBERT) model.
* **reduced_data**: In the "reduced_data" there are all the data obtained after running all the scripts in the scripts folder, in particular in this folder there are the embeddings in their two versions (CNN and AutoEncoder), the corresponding IDs list to know from which image is related each embedding, and the reduced catalogue of outfits (only those that contains the items belonging to the four chosen categories).
* **Images_categorized**: The "Images_categorized" folder contains all the images divided into three folds, named 'train', 'val' and 'test'. The 'train' folder contains the training set and the 'val' folder contains the validation set on which accuracy is measured. The 'test' folder is used for testing the pre-trained model. The structure within the 'train', 'val' and 'test' folders will be the same. They both contain one folder per class. All the images of that class are inside the folder named by class name.
* **dataset_catalogue**: The "dataset_catalogue" folder contains all the images that are present in only the outfits have taken into consideration after the data preparation step, that is, all the images present in the "Images_categorized" folder but not structured in sub-folders.
* **trainings**: The "trainings" folder contains the Python scripts that perform the training of the two models used to obtain the embeddings.
* **utility**: The "utility" folder contains all the necessary utility functions to make the data preparation and the training work. These functions are imported and used in the other Python files.

## umBERT (aUtfit coMpletion with BERT-like model)
The architecture, BERT-like, is composed of a stack of encoders like the one described in the Transformer paper and implemented through PyTorch libraries.
The model is trained on two different phase (pre-training, fine-tuning):
* In **pre-training**, the main objective is to reconstruct the embeddings of the input sequence.
  To achieve this, four straightforward decoders are added on top of the last TransformerEncoder, taking the four encoded embeddings as their input.
  The primary goal of these decoders is to produce outputs that closely resemble the original embeddings fed into the TransformerEncoder stack.
  To measure the similarity between the decoder outputs and the input embeddings, the Mean Squared Error (MSELoss) is used as the loss function.
  This loss helps optimize the model to reconstruct the embeddings effectively during pre-training.
* During the **fine-tuning** process, one item from the input sequence is intentionally masked or missing.
  The primary objective is to produce the correct and complete sequence, which involves accurately reconstructing not only the embeddings of non-masked items but also the embedding corresponding to the masked item.
  The fine-tuning process aims to optimize the model to effectively reconstruct the entire sequence with the masked item accurately filled in.
  Similar to pre-training, the Mean Squared Error (MSELoss) is utilized as the loss function to measure the discrepancy between the predicted embeddings and the ground truth during fine-tuning.
  This ensures the model learns to reconstruct the missing item as accurately as possible.

## The inputs of umBERT
The input of the model is a sequence of four items (shoes, top, accessory, bottom).
The items are represented by their embeddings, which are obtained using two different models:

### Pre-trained CNN (resnet18)
The first model is a pre-trained CNN (resnet18 trained on ImageNet dataset) that takes as input an image and outputs a 128-dimensional embedding.
The model is fine-tuned on the Polyvore dataset, which contains images of items belonging to the categories ['tops','bottoms','accessories','shoes'].

Before fine-tuning the model, the last fully connected layer is replaced with a new fully connected layer with the given number of output features (the desired size of the embeddings) 
and a final fully connected layer with the given number of classes to classify (bottoms, tops, shoes, accessories) is added.

### AutoEncoder (https://github.com/VainF/pytorch-msssim/tree/master)
The AutoEncoder model used is trained with a loss based on the Structured Similarity Index (SSI).
The SSI is a similarity measure between two images that takes into account the structural information of the images, thus is more suitable than MSE in capturing structured similarity between two images

### Comparison between the two models
The two embeddings obtained from the two models are compared to see which one is better.
The embeddings of the Autoencoder seem to be better than the embeddings of the pre-trained CNN in the task of outfit completion
because they represent better the similarities between similar items (e.g. two different shoes of the same brand).
Thus, the embeddings of the Autoencoder are used as input of the umBERT model.

## Preparation of the dataset
The dataset used is the Polyvore outfits downloadable from the following link: https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/

Once the dataset has been downloaded and added to the root folder of the project as "dataset" folder, the scripts in the "scripts" folder must be executed in the following order:
* **0_data_preparation.py**:
  The Polyvore outfits contains a lot of category of items, this script takes only the items belonging to the categories shoes, tops, accessories and bottoms and create a new catalogue of items.
* **1_0_compatibility_data_preparation.py**:
  This script starting from txt files about compatible outfits (compatibility_<train,valid,test>.txt), obtain a dataframe containing outfit composed of 4 items belonging to the categories ['tops','bottoms','accessories','shoes'] and save them into a csv file containing the item IDs.
* **1_1_create_catalogue_dataset.py**:
  This script takes images from a catalogue and move them in the folder "dataset_catalogue". Note that the folder "dataset_catalogue" must be created before running this script.
* **1_2_split_and_organize_dataset.py**:
  This script takes a dataset of images, splits it into train, val and test sets and organizes them in folders with the name of the label associated with each image.
* **2_AE_compute_embeddings.py**:
  This script obtains the embeddings of the dataset using the trained model trained_fashion_VAE_resnet18_128.pth in the "checkpoints" folder.
* **2_resnet_compute_embeddings.py**:
  This script obtains the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18_128.pth in the "checkpoints" folder.

## WumBERT
WumBERT is a modified version of umBERT, the main change is in the embeddings it takes as inputs. 
In fact, the intuition behind this model, and the main difference with the umBERT model, is that here the embeddings of the items are not fixed, but they are initialised with the embeddings returned by the AutoEncoder, and then they are learned during the training process of the model.
