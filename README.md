# MachineTranslation
Machine Translation Using Recurrent Neural Networks

## Description
The aim of this project is to train a machine translation model, which takes natural language instructions and converts them into a machine readable format. The data is a sample from survey questionnaires, where each question is asked based on a specific answer selected in some previous question. The training data has three columns:
1. QID: Question ID for the current question and questions IDs that current question id depenent.
2. CONDITION: condition stated in natural longuage.
3. OUTPUT: Machine readbale condition target output.

Here is an example training data sample:
```
QID:  QE2,hQK
CONDITION:  IF QE2 = NOT 99, CONTINUE, OTHERWISE END AT hQK
OUTPUT:  QE2.notany(99) 
```

## Designed Neural ModelModel overview:
A  sequence-to-sequence (seq2seq) model is trainined using Recurrent Neural Networks (RNNs) with with Long Short Term Memory (LSTM).

The model has an Encoder cosistsing of LSTM layers to extract latent representation of the input sequence. The Decoder consists of LSTM layers followed by a Fully-connected layer. Finally a Softmax Activation function is used to predict the different sequence words probabilities of each time step of target sequence.

To train the network, the input sequence is fed to Encoder. The encoder hidden state and the target sequence (pre-padded by start-of-sequence token) is fed to the decoder. The loss function id defined such that Decoder must predict the target sequence one step ahead of being fed to Decoder.

During prediction, the encoder hidden state and a start-of-sequence token is fed to the decoder, the first predicted sequence by decoder is then being fed back to it for the next step. This procedure will continue until a special token (end-of-sequence token) is predicted or maximum sequence length is reached.

To increase the prediction accuracy of the model a beam search algorithms is used.


## How to use the model
### Train the model:

Model can be trained by running the python script train.py. The required packages are included in “requirement.txt”
```
$ pip3 install requirements.txt
$ python3 train.py
```
### Predict using the model
Model can be used for prediction of the Test Data by running the python script predict.py 
```
$ python3 predict.py
```
The model output predicstions in shel and writes the outputs to a separate file in test data path.

## Model Performance
The trained model provides accuracy of about 95% on the validation set which is remarkable considering that train dataset is not large.
The following are some of the model predictions on training data:

```
Sample index 0
QID:  Q26,Q20
CONDITION:  ASK Q26 IF Q20 = 0
OUTPUT:  Q20.any(0) 
Predicted OUTPUT:  q20.any(0) 
Sample index 10
QID:  8012,8042,8050
CONDITION:  IF 8012= 1,2,3 or 4 AND 8042=1,2 or 3 AND 8050=1 then classify as subgroup A
OUTPUT:  8012.any(1,2,3,4)&8042.any(1,2,3)&8050.any(1) 
Predicted OUTPUT:  8012.any(1,2,4,3,4, 
Sample index 20
QID:  QE2,hQK
CONDITION:  IF QE2 = NOT 99, CONTINUE, OTHERWISE END AT hQK
OUTPUT:  QE2.notany(99) 
Predicted OUTPUT:  qe2.notany(99) 
```

## Discussion
Beam search is used to make a better prediction on test data. Thus, the model makes several prediction and chooses the best possible sequence. However, this mechanism slows down prediction. For faster prediction number of beam search branches should be set to 1 (not use beam search).
The model performance can be further enhanced by:
1. Training on a larger train data and increasing model size
2. Further optimization of the model parameters
3. Using Embedding layer at Encoder and Decoder inputs
4. Using Self Attention Based Models instead of RNN (such as BERT and GPT-2)
