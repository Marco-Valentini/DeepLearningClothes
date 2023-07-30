# DeepLearningClothes
This repository contains a BERT-like model that can accomplish the fill-in-the-blank task in the fashion domain.
The model will take a sequence of four items (shoes, top, accessory, bottom) with one missing (masked) and predict the missing item given the context.

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
The AutoEncoder model is composed of an encoder and a decoder.


The two embeddings obtained from the two models are compared to see which one is better.
The embeddings of the Autoencoder seem to be better than the embeddings of the pre-trained CNN in the task of outfit completion
because they represent better the similarities between similar items (e.g. two different shoes of the same brand).
Thus the embeddings of the Autoencoder are used as input of the umBERT model.

## Preparation of the dataset
The dataset used is the Polyvore oufits downloadable from the following link: https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/

Once the dataset has been downloaded and added to the root folder of the project as "dataset" folder, the scripts in the "scripts" folder must be executed in the following order:
* **0_data_preparation.py**:
  The Polyvore oufits contains a lot of category of items, this script takes only the items belonging to the categories shoes, tops, accessories and bottoms and create a new catalogue of items.
* **1_0_compatibility_data_preparation.py**:
  This script starting from txt files about compatible outfits (compatibility<train,valid,test>.txt), obtain a dataframe containing outfit composed of 4 items belonging to the categories ['tops','bottoms','accessories','shoes'] and save them into a csv file containing the item IDs.
* **1_1_create_catalogue_dataset.py**:
  This script takes images from a catalogue and move them in the folder 'dataset_catalogue'. Note that the folder 'dataset_catalogue must be created before running this script.
* **1_2_split_and_organize_dataset.py**:
  This script takes a dataset of images, splits it into train, val and test sets and organises them in folders with the name of the label associated with each image.
* **2_AE_compute_embeddings.py**:
  This script obtains the embeddings of the dataset using the trained model trained_fashion_VAE_resnet18_128.pth in the "checkpoints" folder.
* **2_resnet_compute_embeddings.py**:
  This script obtains the embeddings of the dataset using the fine-tuned model finetuned_fashion_resnet18_128.pth in the "checkpoints" folder.

## WumBERT
WumBERT is a modified version of umBERT, the main change is the embeddings it takes as inputs. 
In fact, the intuition behind this model, and the main difference with the umBERT model, is that here the embeddings of the items are not fixed, but they are learned during the training process of the model.
