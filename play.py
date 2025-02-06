import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re as re

from model import DescriptionWriter
from get import get_columns

input_size = 93
hidden_size = 256 * 2
output_size = 2100
num_layers = 3

best_model = None

try:
    saved_data = torch.load("models/FINAL_AI_BUILD.pth", weights_only=True)
    best_model = saved_data[0]
    word_to_int = saved_data[1]
    input_size = saved_data[2]
    hidden_size = saved_data[3]
    output_size = saved_data[4]
    num_layers = saved_data[7]
except FileNotFoundError:
    pass

model = DescriptionWriter(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size, 
    num_layers=num_layers,    
)

if best_model != None:
    model.load_state_dict(best_model)
    int_to_word = {i: c for c, i in word_to_int.items()}
model.eval()

def get_response(price: int, sqft: int, bed: int, bath: int, acres: float, exterior: str, heating: str, subdivsion: str, parking: str, description: str):
    price = price / 6000
    sqft = sqft / 100
    acres = acres * 100

    words = 50

    def get_word_to_int(s: str, des:bool=False):
        s = s.lower()
        s = s.replace(".", " thisistheendofthesentenceword ")
        s = s.replace(",", " acommashouldgohere")
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip() 
        s = s.split(" ")
        res = []
        if des:
            for _ in range(words - len(s)):
                res.append(0)

        for w in s:
            res.append(word_to_int[w])

        return res

    def get_max_words(column_name: str) -> int:
        df = pd.read_csv("C:\Python Code\Real Estate\data copy.csv")

        max_words = 0

        for i in df[column_name]:
            max_words = max(len(get_word_to_int(str(i))), max_words)

        return max_words

    def gather_tensor_data():
        df = pd.DataFrame()

        ex_max = 20
        pk_max = 21
        he_max = 20
        sub_max = 7

        ex = pd.DataFrame(np.zeros([1, ex_max]), columns=[f"ExteriorAsInt_{str(i)}" for i in range(ex_max)])
        pk = pd.DataFrame(np.zeros([1, pk_max]), columns=[f"ParkingAsInt_{str(i)}" for i in range(pk_max)])
        he = pd.DataFrame(np.zeros([1, he_max]), columns=[f"HeatingAsInt_{str(i)}" for i in range(he_max)])
        sub = pd.DataFrame(np.zeros([1, sub_max]), columns=[f"SubAsInt_{str(i)}" for i in range(sub_max)])
        
        for column in get_columns():
            col = pd.DataFrame(np.zeros([1, 1]), columns=[column])
            df = df.join(col)
        
        df = df.drop("Exterior", axis=1).drop("Parking", axis=1)\
            .drop("Heating", axis=1).drop("Description", axis=1)\
            .drop("Subdivision", axis=1)

        df.loc[0, "Price"] = price
        df.loc[0, "SQFT"] = sqft
        df.loc[0, "Beds"] = bed
        df.loc[0, "Baths"] = bath
        df.loc[0, "Acres"] = acres

        df = df.join(ex).join(pk).join(he).join(sub)

        for i, j in enumerate(get_word_to_int(heating)):
            df.loc[0, "HeatingAsInt_" + str(i)] = j

        for i, j in enumerate(get_word_to_int(parking)):
            df.loc[0, "ParkingAsInt_" + str(i)] = j

        for i, j in enumerate(get_word_to_int(exterior)):
            df.loc[0, "ExteriorAsInt_" + str(i)] = j
        
        for i, j in enumerate(get_word_to_int(subdivsion)):
            df.loc[0, "SubAsInt_" + str(i)] = j

        for i, j in enumerate(get_word_to_int(description, True)):
            if i < words:
                col = pd.DataFrame(np.zeros([1, 1]), columns=["des_" + str(i)])
                df = df.join(col)
                df.loc[0, "des_" + str(i)] = j

        c = []
        for n, _ in df.items():
            c.append(n)


        df = df.fillna(0).apply(pd.to_numeric)

        return df


    def convert_df_to_tensor(t: pd.DataFrame):
        return torch.tensor(t.to_numpy(), dtype=torch.long)

    def shift_des(new_word: int, df: pd.DataFrame):
        for i in range(words-1):
            df.loc[0, "des_" + str(i)] = df.loc[0, "des_" + str(i+1)]
        df.loc[0, f"des_{words-1}"] = new_word
        return df

    def play():
        try:
            t = gather_tensor_data()
            tensor = convert_df_to_tensor(t)
            sentence_length = 100
        except KeyError as e:
            return f"An Error Occured: {e} is not a word in our training data"

        out = ""

        with torch.no_grad():
            for i in range(sentence_length):
                output = model(tensor)
                predicted_index = torch.argmax(output, dim=-1).item() # Reasearch this
                word = int_to_word[predicted_index]

                Cap = False

                if i > 2:
                    if out[-2] == ".":
                        Cap = True
                elif i == 0:
                    Cap = True
                if Cap and word != "acommashouldgohere":
                    word = word[0].capitalize() + word[1:]
                out += word + " "

                out = out\
                    .replace("acommashouldgohere", ",")\
                    .replace(" thisistheendofthesentenceword", ".")\
                    .replace(" acommashouldgohere", ",")

                if word == "thisistheendofthesentenceword":
                    pass

                out = out.replace(" ,", ",").replace(" .", ".")

                t = shift_des(predicted_index, t)
                tensor = convert_df_to_tensor(t)

        return description + " " + out

    return play()

'''
1. load model
2. get the word to int and reverse it
3. load model variables into a new model
4. turn the words you want it to predict into a tensors
5. run it 

'''