import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_model='lstm', n_layer=1):
        super(CharRNN, self).__init__()

        # 변수 정의
        self.rnn_model = rnn_model.lower() # 모델 구분
        self.input_size = input_size # input char의 vector size. 즉, 한글 dataset의 크기
        self.hidden_size = hidden_size # hidden layer에서의 vector size
        self.output_size = output_size # output의 size (CharRNN은 input set과 output set이 동일하여 size가 동일할 듯)

        self.n_layer = n_layer # for what?

        # 모델 구성하는 데 필요한 부분
        # nn.Embedding을 통해서 계산을 줄였음.
        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.rnn_model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layer)
        elif self.rnn_model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layer)
        self.decoder = nn.Linear(hidden_size, output_size, bias=True)


    def forward(self, _input, hidden):
        # _input은 [batch size, 자모분리된 한글 seq -> (len(글자길이) * 3)]
        batch_size = _input.size(0) # text length

        # hidden_size 만큼의 dim을 가지는 vector 로 변경되었음.
        # 차원 변화 : [batch_size, 자모 분리 seq] -> [batch_size, 자모 분리 seq, embedding 사이즈]
        encoded = self.encoder(_input) 

        # gru의 경우 init hidden이 1개
        # lstm의 경우 init hidden이 hidden과 cell로 2개
        output, hidden = self.rnn(encoded, hidden)

        # 다시 embdding 차원에서 hangul_set으로
        output = self.decoder(output)

        return output, hidden
        
    def init_hidden(self, batch_size):
        #for lstm
        if self.rnn_model == "lstm":
            return (Variable(torch.zeros(self.n_layer, batch_size, self.hidden_size)).to(device),
                    Variable(torch.zeros(self.n_layer, batch_size, self.hidden_size)).to(device))
        
        #for gru
        return Variable(torch.zeros(self.n_layer, batch_size, self.hidden_size)).to(device)