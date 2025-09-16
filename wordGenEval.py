import torch
import torch.nn as nn
import numpy as np


# A Class To Pre-Process The Data
class Data_PreProcessing:
    def __init__(self, list_sentences):
        self.list_sentences = list_sentences
        self.one_paragraph = ''
        self.list_all_words = []
    
    # Convert the list sentences to one paragraph
    def to_paragraph(self):
        self.one_paragraph = ' '.join(self.list_sentences)
        return self.one_paragraph

    # Encoding words to index, decoding index to words, return 
    def encode_decode_dictionaries(self):
        if self.one_paragraph == '' and len(self.list_sentences) > 0:
            self.to_paragraph()
        self.list_all_words = [word for word in self.one_paragraph.split()]
        wti = {w:i for i, w in enumerate(set(self.list_all_words))}
        itw = {i:w for i, w in enumerate(set(self.list_all_words))}
        return wti, itw, len(wti)

class Embedding:
    def __init__(self, n_words, vector_size = 64):
        self.vector_size = vector_size
        self.n_words = n_words
        self.embedded_vector = np.random.random((self.n_words, self.vector_size))
    

    def get_vectors(self, indices):
        return self.embedded_vector[indices, :]


# Define the model architecture again (same as before)
class WordGen(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(WordGen, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, out_size)
    
    def forward(self, x, hidden=None):
        if hidden is not None:
            x, hidden = self.rnn(x, hidden)
        else:
            x, hidden = self.rnn(x)
        
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x, hidden
    
    def zero_hidden_state(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# text_data = "Mohamed Saad Is A Good Boy .Mohamed Almorsi Is A Good Boy .".split(),
wti, itw = torch.load('./data/wti_dict.pth', weights_only=True), torch.load('./data/itw_dict.pth', weights_only=True)
n_words = len(wti)
# Parameters (must be the same as in training!)
hidden_size = 128
embedded_vector_size = 64
out_size =  len(wti)   # or load from preprocessing again
emb = Embedding(out_size, embedded_vector_size)
embedded_words = np.load('./data/embedded_vector.npy')
emb.embedded_vector = embedded_words


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = WordGen(embedded_vector_size, hidden_size, out_size)

# Load trained weights
model.load_state_dict(torch.load("./model/word_eval_model.pth", map_location=device, weights_only=True))
model.to(device)

def generate(prefex:list, n_predicted_words):
    model.eval()
    hidden = model.zero_hidden_state(device).squeeze(0)
    for word in prefex:
        input_vec = emb.get_vectors(wti[word])
        out, hidden = model(torch.tensor(input_vec.reshape(1, -1), dtype=torch.float32, device=device), hidden)


    for i in range(n_predicted_words):
        probs = torch.softmax(out, dim=1)
        pred_index = np.random.choice(n_words, p=probs.squeeze(0).detach().cpu().numpy())
        pred_word = itw[pred_index]
        print(pred_word, end=' ')

        # if pred_index == wti['.']:
        #     break

        input_vec = emb.get_vectors(wti[pred_word])
        out, hidden = model(torch.tensor(input_vec.reshape(1, -1), dtype=torch.float32, device=device), hidden)
        




prefex = ['make', 'it']

generate(prefex, 100)