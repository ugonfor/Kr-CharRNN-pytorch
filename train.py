import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook
from torch.autograd import Variable

import random
import time

from model import CharRNN
from preprocess import hangul_int_tensor, hangul_num, hangul_set
from generation import generate_text
#https://github.com/bluedisk/hangul-toolkit
#https://github.com/spro/char-rnn.pytorch


#load hangul Dataset
hangul_data = open("data.txt", "rt", encoding='utf-8')
hangul_data = hangul_data.read()

# variable set
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 50
chunk_len = 70

#model
model = CharRNN(input_size=hangul_num, hidden_size=10, output_size=hangul_num).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# _chunk_len : 글자 개수 (문장 단위)
# _batch_size : 문장의 개수
def generate_traning_set(_chunk_len, _batch_size):
    _input = torch.LongTensor(_batch_size, 3*_chunk_len)
    _target = torch.LongTensor(_batch_size, 3*_chunk_len)

    for idx in range(_batch_size):
        _start_index = random.randint(0, len(hangul_data) - _chunk_len)
        _end_index = _start_index + _chunk_len + 1
        chunk = hangul_data[_start_index:_end_index]
        
        #print(hangul_int_tensor(chunk[:-1]).view(-1))
        _input[idx] = hangul_int_tensor(chunk[:-1]).view(-1)
        _target[idx] = hangul_int_tensor(chunk[1:]).view(-1)

    _input = Variable(_input).to(device)
    _target = Variable(_target).to(device)

    return _input, _target


def train(_input, _target):
    hidden = model.init_hidden(batch_size)

    model.zero_grad()
    loss = 0

    for idx in range(chunk_len):
        output, hidden = model(_input[:,idx].view(1,-1), hidden) # batch, RNN
        loss += criterion(output.view(batch_size, -1), _target[:,idx])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data / chunk_len


def main_train(epochs):
    print(f'Device : {device}')
    start = time.time()
    loss_total = 0

    print("Training for %d epochs..." % epochs)
    model.train()
    for epoch in tqdm_notebook(range(1, epochs + 1)):
        loss = train(*generate_traning_set(chunk_len, batch_size))
        loss_total += loss

        print(f'time : {time.time() - start :>10.2f}    epoch : {epoch} ({epoch/epochs * 100:>10.2f} % ) loss : {loss:>10.2f}')
    print(f'Training Done!')

def main_eval():
    print('Evaluate the model')
    model.eval()
    print(generate_text(model, "와", 100))


if __name__ == "__main__" :
    main_train(5)
    main_eval()