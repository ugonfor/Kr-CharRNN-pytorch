import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import CharRNN
from preprocess import hangul_int_tensor, hangul_num, hangul_set
import hgtk 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_text(_model, start_str, predict_len, cuda=False):
    hidden = _model.init_hidden(3) # batch size = 3 (당연히! 자모받힘까지 3개니까)
    start_input = Variable(hangul_int_tensor(start_str).unsqueeze(0))
    start_input = start_input.to(device)
    print(start_str)
    predicted = start_str
    
    #hidden state를 초기 생성
    for p in range(len(start_input) -1):
        _, hidden = _model(start_input[:,p],hidden)
    
    _input = start_input[:,-1]
    for p in range(predict_len):
        output, hidden = _model(_input, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(3,-1).div(0.8).exp() #한 글자 한글자가 70 * 3 이니까.
        #print(output_dist)
        top_1 = torch.multinomial(output_dist[0], 1)[0] # most significant value ! 자음
        top_2 = torch.multinomial(output_dist[1], 1)[0] # most significant value ! 모음
        top_3 = torch.multinomial(output_dist[2], 1)[0] # most significant value ! 받힘
         
        #print(top_1)
        #print(top_2)
        #print(top_3)
        
        # Add predicted character to string and use as next input   
        predicted_1 = hangul_set[top_1] # 자음
        predicted_2 = hangul_set[top_2] # 모음
        predicted_3 = hangul_set[top_3] # 받힘

        try:
            predicted_char = hgtk.letter.compose(*[predicted_1, predicted_2, predicted_3])
        except hgtk.exception.NotHangulException as e:
            predicted_char = predicted_1 + predicted_2 + predicted_3

        predicted += predicted_char

        _input = Variable(torch.tensor([top_1,top_2,top_3]).unsqueeze(0))
        _input = _input.to(device)

    return predicted
