from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import re # for regular expression
import pandas as pd
from collections import Counter

app = FastAPI() # Creating an instance of FAST API
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Define a Pydantic model for the request body
class InputSentence(BaseModel):
    sentence: str

@app.get("/") # Get method is to Get some response from the Server
def greet():
    return {"Hello World!"} # Return the JSON object representing Hello World

# Define the route for your API
@app.post("/predict")
async def predict_next_word(input_data: InputSentence):
    csv_file_path = '/Users/pc/Desktop/Backend/clean_date_categories.csv'
    column_name = 'text'
    # Read only the specified column for the nrows number of rows
    first_row_text = pd.read_csv(csv_file_path, usecols=[column_name], nrows=63050)
    # Read 'Bichar' and 'Sahityarkala' data from row 5600 to 7500
    # blog_news = first_row_text.iloc[26344:26748, first_row_text.columns.get_loc('text')] # Blog
    # sampadakiya_news1 = first_row_text.iloc[38090:38910, first_row_text.columns.get_loc('text')] # Sampadakiya set 1
    # sampadakiya_news2 = first_row_text.iloc[22750:23330, first_row_text.columns.get_loc('text')] # Sampadakiya set 2

    # Concatenate the subsets into one variable
    # subset_of_text = pd.concat([blog_news, sampadakiya_news1, sampadakiya_news2], ignore_index=True)
    # subset_of_text = pd.concat([blog_news], ignore_index=True)
    subset_of_text = first_row_text.iloc[5600:5700, first_row_text.columns.get_loc('text')] # Satyasa did on this

    # Remove '\n' from the text column
    subset_of_text = subset_of_text.str.replace('\n', ' ')
    # Convert subset_of_text to string
    subset_of_text_string = subset_of_text.str.cat(sep=' ')
    # Remove Non-Nepali characters
    subset_of_text_string = re.sub('[A-Za-z]+', ' ', subset_of_text_string)


    # Tokenize Nepali text based on whitespace characters
    # this function always return list
    nepali_words = re.findall(r'\S+', subset_of_text_string)


    #set contains unique element
    unique_words = list(set(nepali_words))
    # word_to_index is a dictionary where every unique word is given a particular index
    word_to_index = {word: index for index, word in enumerate(unique_words)}

    word_tensors = [torch.tensor(word_to_index[word]) for word in nepali_words]

    # Define a PyTorch Dataset class for handling text data
    class TextDataset(torch.utils.data.Dataset):
        # Constructor initializes the dataset with arguments (args)
        def __init__(self, args):
            # Store the arguments for later use
            self.args = args
            # Load words from the dataset
            self.words = self.load_words()
            # Get unique words and create mappings
            self.unique_words = self.get_unique_words()

            # Create a mapping from index to word
            self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
            # Create a mapping from word to index
            self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

            # Create a list of word indices based on the mappings
            self.word_indexes = [self.word_to_index[w] for w in self.words]

        # Function to load words from a CSV file
        def load_words(self):
            return nepali_words

        # Function to get unique words and their frequencies
        def get_unique_words(self):
            # Use Counter to count the occurrences of each word in the dataset
            word_counts = Counter(self.words)
            # Return a list of unique words, sorted by their frequencies in descending order
            return sorted(word_counts, key=word_counts.get, reverse=True)

        # Function to get the length of the dataset
        def __len__(self):
            # Return the length of the list of word indexes minus the specified args
            return len(self.word_indexes) - self.args

        def __getitem__(self, index):
            return (
                torch.tensor(self.word_indexes[index:index + self.args]),
                torch.tensor(self.word_indexes[index + 1:index + self.args+ 1])
            )

    class LSTMModel(nn.Module):
        def __init__(self, dataset):
            super(LSTMModel, self).__init__()
            #number of hidden units or neurons in each LSTM layer
            self.lstm_size = 128
            self.embedding_dim = 128
            self.num_layers = 3
            # self.num_layers = 2

            n_vocab = len(dataset.unique_words)
            #embedding layer is necessary to convert word into dense vector
            self.embedding = nn.Embedding(
                #how many unique words
                num_embeddings=n_vocab,
                #what dimension of vector will represent the unique words
                embedding_dim=self.embedding_dim,
            )
            self.lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.lstm_size,
                num_layers=self.num_layers,
                #regularization technique that helps to control overfitting
                dropout=0.2,
            )
            #fully connected layer
            self.fc = nn.Linear(self.lstm_size, n_vocab)

        def forward(self, x, prev_state):
        #applies wrod embedding to input sequence x
            embed = self.embedding(x)
            output, state = self.lstm(embed, prev_state)
            # logits represent how likely each word is to come next in the sequence based on the learned patterns.
            logits = self.fc(output)

            return logits, state


        def init_state(self, sequence_length):
            return (
                torch.zeros(self.num_layers,
                            sequence_length, self.lstm_size),
                torch.zeros(self.num_layers,
                            sequence_length, self.lstm_size)
            )

    # Hyperparameters
    sequence_length = 10
    # batch_size = 64
    # learning_rate = 0.01
    # num_epochs = 100

    # Create the dataset
    dataset = TextDataset(sequence_length)

    # Load your pre-trained LSTM model
    model = LSTMModel(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("/Users/pc/Desktop/Backend/epoch-99.pt", map_location=torch.device(device)))
    model.eval()

    # Preprocess the input sentence
    input_sentence = input_data.sentence
   # Filter out words not in the dataset
    input_words = [word for word in input_sentence.split() if word in dataset.word_to_index]

    # Convert remaining words into indexes
    # input_indexes = [dataset.word_to_index[word] for word in input_words]

    # # Convert indexes into a tensor
    # input_tensor = torch.tensor(input_indexes, dtype=torch.long).unsqueeze(0)
    # # Generate the next word
    # hidden = model.init_state(len(input_indexes))
    # outputs, _ = model(input_tensor, hidden)
    # predicted_index = torch.argmax(outputs[0, -1, :]).item()
    # predicted_word = dataset.index_to_word[predicted_index]
    input_indexes = [dataset.word_to_index[word] for word in input_sentence.split() if word in dataset.word_to_index]

    # model.load_state_dict(torch.load("/content/drive/MyDrive/Minor-Project/epoch-1.pt"))
    # Initialize hidden state
    hidden = model.init_state(1)
   

    # Process each word in the input sentence sequentially
    for input_index in input_indexes:
        # Convert input_index to tensor
        input_tensor = torch.tensor([[input_index]], dtype=torch.long)

        # Forward pass through the model
        outputs, hidden = model(input_tensor, hidden)

    # Get the predicted index of the next word

    predicted_index = torch.argmax(outputs[0, -1, :]).item()
    predicted_word = dataset.index_to_word[predicted_index]    

    return {"input_sentence": input_sentence, "predicted_word": predicted_word}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
