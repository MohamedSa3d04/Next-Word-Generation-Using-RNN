import torch
import torch.nn as nn
import numpy as np

# A Class To Pre-Process The Data
class Data_PreProcessing:
    def __init__(self, text):
        self.one_paragraph = text
        self.list_all_words = []

    # Encoding words to index, decoding index to words, return 
    def encode_decode_dictionaries(self):
        assert self.one_paragraph != ''

        self.list_all_words = [word for word in self.one_paragraph.split()]
        wti = {w:i for i, w in enumerate(set(self.list_all_words))}
        itw = {i:w for i, w in enumerate(set(self.list_all_words))}
        return wti, itw, len(wti)


# A Class To Get Embedding Vectors of The Word
class Embedding:
    def __init__(self, n_words, vector_size = 64):
        self.vector_size = vector_size
        self.n_words = n_words
        self.embedded_vector = np.random.random((self.n_words, self.vector_size))
    

    def get_vectors(self, indices):
        return self.embedded_vector[indices, :]


class WordGen(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(WordGen, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, out_size)
    
    def forward(self, x, hidden=None):
        if hidden:
            x, hidden = self.rnn(x, hidden)
        else:
            x, hidden = self.rnn(x)
        
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x, hidden
    
    def zero_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
with open('dataText.txt', 'r') as file:
    text_data = file.read()

data_proccesor = Data_PreProcessing(text_data)
wti, itw, n_words = data_proccesor.encode_decode_dictionaries()

text_data = text_data.split()
hidden_size = 128
embedded_vector_size = 64
out_size = n_words
emb = Embedding(n_words, embedded_vector_size)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = WordGen(embedded_vector_size, hidden_size, out_size)   
model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
n_epocs = 500
sequence_len = n_words - 1

for epoch in range(n_epocs):
    total_loss = 0
    count = 0
    for i in range(0, len(text_data) - sequence_len + 1, sequence_len):
        hidden = model.zero_hidden_state()
        from_, to_ = i, i+sequence_len
        sequence = text_data[from_:to_]
        for i, word in enumerate(sequence):
            
            target_index = torch.tensor(wti[text_data[i+1]], dtype=torch.long, device=device)
            input_vec = emb.get_vectors(wti[word])

            out, hidden = model(torch.tensor(input_vec.reshape(1, -1), dtype=torch.float32, device=device))
            loss = criterion(out, target_index.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

    print(total_loss / count)





# Save the trained model
np.save('./embedded_vector.npy', emb.embedded_vector)
torch.save(model.state_dict(), "./model/word_eval_model.pth")
torch.save(wti, './data/wti_dict.pth')
torch.save(itw, './data/itw_dict.pth')
print("Model saved successfully!")



