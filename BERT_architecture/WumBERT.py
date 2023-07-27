import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from BERT_architecture.utilities_WumBERT import find_closest_embeddings
import matplotlib.pyplot as plt
from datetime import datetime
import tqdm


class WumBERT(nn.Module):
    def __init__(self, embeddings: np.ndarray, num_encoders, num_heads, dropout, shoes_idx, tops_idx, accessories_idx,
                 bottoms_idx, dim_feedforward=None):
        super(WumBERT, self).__init__()

        self.d_model = embeddings.shape[1]
        if dim_feedforward is None:
            dim_feedforward = 4 * self.d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.shoes_idx = shoes_idx
        self.tops_idx = tops_idx
        self.accessories_idx = accessories_idx
        self.bottoms_idx = bottoms_idx

        # add the random mask to the embeddings
        mask = np.mean(embeddings, axis=0) + np.random.randn(1, self.d_model)
        embeddings = np.concatenate((embeddings, mask), axis=0)
        self.embeddings = nn.Embedding(num_embeddings=embeddings.shape[0], embedding_dim=self.d_model).from_pretrained(
            torch.from_numpy(embeddings).float(), freeze=False)
        self.num_heads = num_heads
        self.num_encoders = num_encoders
        self.encoder_stack = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        ), num_layers=self.num_encoders,
            enable_nested_tensor=(num_heads % 2 == 0))
        self.dec_shoes = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.dec_tops = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.dec_acc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.dec_bottoms = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
            # TODO valuta se aggiungere altri layer
        )

    def forward(self, x):
        return self.encoder_stack(x)

    def forward_reconstruction(self, input_idx):
        input = self.embeddings(input_idx)  # input idx deve essere batch_size x seq_length
        output = self.forward(input)
        rec_shoes = self.dec_shoes(output[:, 0, :])
        rec_tops = self.dec_tops(output[:, 1, :])
        rec_acc = self.dec_acc(output[:, 2, :])
        rec_bottoms = self.dec_bottoms(output[:, 3, :])
        return rec_shoes, rec_tops, rec_acc, rec_bottoms

    def forward_with_masking(self, input_idx):
        mask_ids = torch.tensor(self.embeddings.weight.shape[0] - 1).to(input_idx.device)
        masked_positions = []
        masked_items = []
        for i in range(0, input_idx.shape[0], 4):
            for j in range(input_idx.shape[1]):
                masked_items.append(input_idx[i + j, j].item())
                input_idx[i + j, j] = mask_ids
                masked_positions.append(j)
        input = self.embeddings(input_idx)  # input idx deve essere batch_size x seq_length
        output = self.forward(input)
        rec_shoes = self.dec_shoes(output[:, 0, :])
        rec_tops = self.dec_tops(output[:, 1, :])
        rec_acc = self.dec_acc(output[:, 2, :])
        rec_bottoms = self.dec_bottoms(output[:, 3, :])
        masked_logits = []
        for i in range(len(masked_positions)):
            if masked_positions[i] == 0:
                masked_logits.append(rec_shoes[i])
            elif masked_positions[i] == 1:
                masked_logits.append(rec_tops[i])
            elif masked_positions[i] == 2:
                masked_logits.append(rec_acc[i])
            elif masked_positions[i] == 3:
                masked_logits.append(rec_bottoms[i])
        return rec_shoes, rec_tops, rec_acc, rec_bottoms, torch.stack(masked_logits), masked_positions, masked_items

    def fit_reconstruction(self, dataloaders, device, epochs, criterion, optimizer):
        train_loss = []  # keep track of the loss of the training phase
        val_loss = []  # keep track of the loss of the validation phase
        train_acc_decoding = []  # keep track of the accuracy of the training phase on the MLM reconstruction task
        val_acc_decoding = []  # keep track of the accuracy of the validation phase on the reconstruction task

        valid_loss_min = np.Inf  # track change in validation loss
        early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss
        best_model = deepcopy(self)

        for epoch in range(epochs):
            for phase in ['train', 'val']:
                print(f'Reconstruction pre-training epoch {epoch + 1}/{epochs} | Phase {phase}')
                if phase == 'train':
                    self.train()
                    print("Training...")
                else:
                    self.eval()
                    print("Validating...")
                running_loss = 0.0
                accuracy_shoes = 0.0
                accuracy_tops = 0.0
                accuracy_acc = 0.0
                accuracy_bottoms = 0.0
                for inputs in tqdm.tqdm(dataloaders[phase], colour='blue'):
                    inputs = inputs.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        rec_shoes, rec_tops, rec_acc, rec_bottoms = self.forward_reconstruction(inputs)
                        loss = criterion(rec_shoes, self.embeddings(inputs)[:, 0, :]) + criterion(rec_tops,
                                                                                                  self.embeddings(
                                                                                                      inputs)[:, 1,
                                                                                                  :]) + criterion(
                            rec_acc, self.embeddings(inputs)[:, 2, :]) + criterion(rec_bottoms,
                                                                                   self.embeddings(inputs)[:, 3, :])
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # update the loss value (multiply by the batch size)
                    running_loss += loss.item() * inputs.size(0)
                    # compute the closest embeddings to the reconstructed embeddings
                    pred_shoes = find_closest_embeddings(rec_shoes, self.embeddings(
                        torch.LongTensor(list(self.shoes_idx)).to(device)), self.shoes_idx)
                    pred_tops = find_closest_embeddings(rec_tops, self.embeddings(
                        torch.LongTensor(list(self.tops_idx)).to(device)), self.tops_idx)
                    pred_acc = find_closest_embeddings(rec_acc, self.embeddings(
                        torch.LongTensor(list(self.accessories_idx)).to(device)), self.accessories_idx)
                    pred_bottoms = find_closest_embeddings(rec_bottoms, self.embeddings(
                        torch.LongTensor(list(self.bottoms_idx)).to(device)), self.bottoms_idx)

                    # update the accuracy of the reconstruction task
                    accuracy_shoes += np.sum(np.array(pred_shoes) == inputs[:, 0].cpu().numpy())
                    accuracy_tops += np.sum(np.array(pred_tops) == inputs[:, 1].cpu().numpy())
                    accuracy_acc += np.sum(np.array(pred_acc) == inputs[:, 2].cpu().numpy())
                    accuracy_bottoms += np.sum(np.array(pred_bottoms) == inputs[:, 3].cpu().numpy())

                    # compute the average loss of the epoch
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                # compute the average accuracy of the MLM task of the epoch
                epoch_accuracy_shoes = accuracy_shoes / len(dataloaders[phase].dataset)
                epoch_accuracy_tops = accuracy_tops / len(dataloaders[phase].dataset)
                epoch_accuracy_acc = accuracy_acc / len(dataloaders[phase].dataset)
                epoch_accuracy_bottoms = accuracy_bottoms / len(dataloaders[phase].dataset)
                epoch_accuracy_reconstruction = (
                                                        epoch_accuracy_shoes + epoch_accuracy_tops + epoch_accuracy_acc + epoch_accuracy_bottoms) / 4

                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy (shoes): {epoch_accuracy_shoes}')
                print(f'{phase} Accuracy (tops): {epoch_accuracy_tops}')
                print(f'{phase} Accuracy (acc): {epoch_accuracy_acc}')
                print(f'{phase} Accuracy (bottoms): {epoch_accuracy_bottoms}')
                print(f'{phase} Accuracy (Reconstruction): {epoch_accuracy_reconstruction}')

                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc_decoding.append(epoch_accuracy_reconstruction)
                else:
                    val_loss.append(epoch_loss)
                    val_acc_decoding.append(epoch_accuracy_reconstruction)

                    # save model if validation loss has decreased
                    if epoch_loss <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            epoch_loss))
                        print('Validation accuracy in reconstruction of the saved model: {:.6f}'.format(
                            epoch_accuracy_reconstruction))
                        # save a checkpoint dictionary containing the model state_dict
                        checkpoint = {'d_model': self.d_model,
                                      'num_encoders': self.num_encoders,
                                      'num_heads': self.num_heads,
                                      'dropout': self.dropout,
                                      'dim_feedforward': self.dim_feedforward,
                                      'embeddings': self.embeddings.weight.data.cpu().numpy(),
                                      'model_state_dict': self.state_dict()}
                        # save the checkpoint dictionary to a file
                        #now = datetime.now()
                        #dt_string = now.strftime("%Y_%m_%d")
                        #torch.save(checkpoint,
                                  # f'./models/{dt_string}_WumBERT_pre_trained_reconstruction_{self.d_model}.pth')
                        valid_loss_min = epoch_loss  # update the minimum validation loss
                        early_stopping = 0  # reset early stopping counter
                        best_model = deepcopy(self)
                    else:
                        early_stopping += 1  # increment early stopping counter
            if early_stopping == 10:
                print('Early stopping the training')
                break
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.legend()
        plt.title('Loss pre-training (reconstruction task)')
        plt.show()
        plt.plot(train_acc_decoding, label='train')
        plt.plot(val_acc_decoding, label='val')
        plt.legend()
        plt.title('Accuracy (reconstruction) pre-training')
        plt.show()
        return best_model, valid_loss_min

    def find_top_k_closest_embeddings(self, recons_embeddings, masked_positions, device, topk=10):
        closest_embeddings = []
        for i, pos in enumerate(masked_positions):
            if pos == 0:
                distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),
                                        self.embeddings(torch.LongTensor(list(self.shoes_idx)).to(device)))
                idx = torch.topk(distances, topk, largest=False, dim=1).indices  # TODO capire la dim
                idx = idx.tolist()
                closest_embeddings.append([list(self.shoes_idx)[i] for i in idx[0]])
            elif pos == 1:
                distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),
                                        self.embeddings(torch.LongTensor(list(self.tops_idx)).to(device)))
                idx = torch.topk(distances, topk, largest=False, dim=1).indices
                idx = idx.tolist()
                closest_embeddings.append([list(self.tops_idx)[i] for i in idx[0]])
            elif pos == 2:
                distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),
                                        self.embeddings(torch.LongTensor(list(self.accessories_idx)).to(device)))
                idx = torch.topk(distances, topk, largest=False, dim=1).indices
                idx = idx.tolist()
                closest_embeddings.append([list(self.accessories_idx)[i] for i in idx[0]])
            elif pos == 3:
                distances = torch.cdist(recons_embeddings[i, :].unsqueeze(0),
                                        self.embeddings(torch.LongTensor(list(self.bottoms_idx)).to(device)))
                idx = torch.topk(distances, topk, largest=False, dim=1).indices
                idx = idx.tolist()
                closest_embeddings.append([list(self.bottoms_idx)[i] for i in idx[0]])
        return closest_embeddings

    def fit_fill_in_the_blank(self, dataloaders, device, epochs, criterion, optimizer):
        train_loss = []  # keep track of the loss of the training phase
        val_loss = []  # keep track of the loss of the validation phase
        train_hit_ratio = []  # keep track of the accuracy of the training phase on the MLM reconstruction task
        val_hit_ratio = []  # keep track of the accuracy of the validation phase on the reconstruction task

        valid_loss_min = np.Inf  # track change in validation loss
        early_stopping = 0  # counter to keep track of the number of epochs without improvements in the validation loss
        best_model = deepcopy(self)
        for epoch in range(epochs):
            for phase in ['train', 'val']:
                print(f'fine-tuning epoch {epoch + 1}/{epochs} | Phase {phase}')
                if phase == 'train':
                    self.train()
                    print("Training...")
                else:
                    self.eval()
                    print("Validating...")
                running_loss = 0.0
                accuracy_shoes = 0.0
                accuracy_tops = 0.0
                accuracy_acc = 0.0
                accuracy_bottoms = 0.0
                hit_ratio = 0.0
                for inputs in tqdm.tqdm(dataloaders[phase],colour='green'):
                    inputs = inputs.to(device)  # df is a dataframe of IDs
                    inputs = inputs.repeat_interleave(4, dim=0)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        rec_shoes, rec_tops, rec_acc, rec_bottoms, masked_logits, masked_positions, masked_items = self.forward_with_masking(
                            inputs)
                        loss = criterion(rec_shoes, self.embeddings(inputs)[:, 0, :]) + criterion(rec_tops,
                                                                                                  self.embeddings(
                                                                                                      inputs)[:, 1,
                                                                                                  :]) + criterion(
                            rec_acc, self.embeddings(inputs)[:, 2, :]) + criterion(rec_bottoms,
                                                                                   self.embeddings(inputs)[:, 3, :])
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    # update the loss value (multiply by the batch size)
                    running_loss += loss.item() * inputs.size(0)
                    # compute the closest embeddings to the reconstructed embeddings
                    pred_shoes = find_closest_embeddings(rec_shoes, self.embeddings(
                        torch.LongTensor(list(self.shoes_idx)).to(device)), self.shoes_idx)
                    pred_tops = find_closest_embeddings(rec_tops, self.embeddings(
                        torch.LongTensor(list(self.tops_idx)).to(device)), self.tops_idx)
                    pred_acc = find_closest_embeddings(rec_acc, self.embeddings(
                        torch.LongTensor(list(self.accessories_idx)).to(device)), self.accessories_idx)
                    pred_bottoms = find_closest_embeddings(rec_bottoms, self.embeddings(
                        torch.LongTensor(list(self.bottoms_idx)).to(device)), self.bottoms_idx)
                    pred_masked = self.find_top_k_closest_embeddings(masked_logits, masked_positions, device, topk=10)

                    # update the accuracy of the reconstruction task
                    accuracy_shoes += np.sum(np.array(pred_shoes) == inputs[:, 0].cpu().numpy())
                    accuracy_tops += np.sum(np.array(pred_tops) == inputs[:, 1].cpu().numpy())
                    accuracy_acc += np.sum(np.array(pred_acc) == inputs[:, 2].cpu().numpy())
                    accuracy_bottoms += np.sum(np.array(pred_bottoms) == inputs[:, 3].cpu().numpy())

                    # compute the hit ratio
                    for i in range(len(pred_masked)):
                        if masked_items[i] in pred_masked[i]:
                            hit_ratio += 1

                epoch_loss = running_loss / (len(dataloaders[phase].dataset) * 4)
                # compute the average accuracy of the MLM task of the epoch
                epoch_accuracy_shoes = accuracy_shoes / (len(dataloaders[phase].dataset) * 4)
                epoch_accuracy_tops = accuracy_tops / (len(dataloaders[phase].dataset) * 4)
                epoch_accuracy_acc = accuracy_acc / (len(dataloaders[phase].dataset) * 4)
                epoch_accuracy_bottoms = accuracy_bottoms / (len(dataloaders[phase].dataset) * 4)
                epoch_accuracy_reconstruction = (
                                                        epoch_accuracy_shoes + epoch_accuracy_tops + epoch_accuracy_acc + epoch_accuracy_bottoms) / 4
                epoch_hit_ratio = hit_ratio  # / len(dataloaders[phase].dataset)
                print(f'{phase} Loss: {epoch_loss}')
                print(f'{phase} Accuracy (shoes): {epoch_accuracy_shoes}')
                print(f'{phase} Accuracy (tops): {epoch_accuracy_tops}')
                print(f'{phase} Accuracy (acc): {epoch_accuracy_acc}')
                print(f'{phase} Accuracy (bottoms): {epoch_accuracy_bottoms}')
                print(f'{phase} Accuracy (Reconstruction): {epoch_accuracy_reconstruction}')
                print(f'{phase} Hit ratio: {epoch_hit_ratio}')
                print(f'{phase} Hit ratio (normalized): {epoch_hit_ratio / len(dataloaders[phase].dataset) * 4}')

                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_hit_ratio.append(epoch_hit_ratio)
                else:
                    val_loss.append(epoch_loss)
                    val_hit_ratio.append(epoch_hit_ratio)

                    # save model if validation loss has decreased
                    if epoch_loss <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            epoch_loss))
                        print('Validation accuracy in reconstruction of the saved model: {:.6f}'.format(
                            epoch_accuracy_reconstruction))
                        # save a checkpoint dictionary containing the model state_dict
                        checkpoint = {'d_model': self.d_model,
                                      'num_encoders': self.num_encoders,
                                      'num_heads': self.num_heads,
                                      'dropout': self.dropout,
                                      'dim_feedforward': self.dim_feedforward,
                                      'embeddings': self.embeddings.weight.data.cpu().numpy(),
                                      'model_state_dict': self.state_dict()}
                        # save the checkpoint dictionary to a file
                        now = datetime.now()
                        dt_string = now.strftime("%Y_%m_%d")
                        torch.save(checkpoint,
                                   f"./models/WumBERT_FT_NE_{self.num_encoders}_NH_{self.num_heads}_D_{self.dropout:.5f}_LR_{optimizer.param_groups[0]['lr']}_OPT_{type(optimizer).__name__}.pth")
                        valid_loss_min = epoch_loss  # update the minimum validation loss
                        early_stopping = 0  # reset early stopping counter
                        best_model = deepcopy(self)
                    else:
                        early_stopping += 1  # increment early stopping counter
            if early_stopping == 10:
                print('Early stopping the training')
                break
        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.legend()
        plt.title('Loss pre-training (reconstruction task)')
        plt.show()
        plt.plot(train_hit_ratio, label='train')
        plt.plot(val_hit_ratio, label='val')
        plt.legend()
        plt.title('Accuracy (reconstruction) pre-training')
        plt.show()
        return best_model, valid_loss_min