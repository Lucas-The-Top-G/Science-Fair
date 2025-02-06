import pandas as pd
import re
import torch
import numpy as np
import random as rand

df1 = pd.read_csv("C:\Python Code\Real Estate\data copy.csv")

df1 = df1.drop("Address", axis=1)
df1 = df1.drop("Garage", axis=1)
df1 = df1.drop("Type", axis=1)

dweeb = df1

def get_columns():
    global dweeb
    placearr = []
    for name, _ in dweeb.items():
        placearr.append(name)
    return placearr

def get_tensors(cols = False, amount_homes = 0,idx = 0):
    df = df1
    print("Amount of homes: ", amount_homes, " idx: ", idx)
    wordArr = set()

    def add_to_arr(cfd: pd.Series, arr: set):
        for ex in cfd:
            ex = str(ex).lower()

            ex = ex.replace(",", " COMMMAAAAAA ") 
            
            ex = re.sub(r"[^\w\s]", " ", ex)
            ex = re.sub(r"\s+", " ", ex).strip() 
            
            exArr = ex.split(" COMMA")
            for word in exArr:
                word = word.replace("COMMMAAAAAA", ",")
                for w in word.split(","):
                    arr.add(w.strip())
        return arr

    wordArr.add("")
    wordArr.add("thisistheendofthesentenceword")
    wordArr.add("acommashouldgohere")

    add_to_arr(df["Subdivision"], wordArr)
    print(wordArr)
    add_to_arr(df["Heating"], wordArr)
    add_to_arr(df["Parking"], wordArr)
    add_to_arr(df["Exterior"], wordArr)
    add_to_arr(df["Description"], wordArr)

    wordArr = sorted(list(wordArr))
    word_to_int = dict((c, i) for i, c in enumerate(wordArr))

    exteriorData = pd.DataFrame()
    parkingData = pd.DataFrame()
    heatingData = pd.DataFrame()
    descriptionData = pd.DataFrame()

    def add_to_dataframe(cfd: pd.Series, n: str, chars = False) -> pd.DataFrame:
        r = pd.DataFrame(columns=[n + "AsInt"])
        for i, ex in enumerate(cfd):
            arr = []
            ex = str(ex).lower()

            if n == "Description":
                ex = ex.replace(".", " thisistheendofthesentenceword")
                ex = ex.replace(",", " acommashouldgohere")
            
            ex = re.sub(r"[^\w\s]", " ", ex)
            ex = re.sub(r"\s+", " ", ex).strip() 
            if not chars:
                exArr = ex.split(" ")
                for word in exArr:
                    arr.append(word_to_int.get(word, 0))
                for i in range(20 - len(exArr)):
                    arr.append(0)
            r.loc[i] = [arr]
        return r

    exteriorData = add_to_dataframe(df["Exterior"], "Exterior")
    parkingData = add_to_dataframe(df["Parking"], "Parking")
    heatingData = add_to_dataframe(df["Heating"], "Heating")
    descriptionData = add_to_dataframe(df["Description"], "Description")
    subdivsion = add_to_dataframe(df["Subdivision"], "Subdivision")

    df = df.drop("Exterior", axis=1).drop("Parking", axis=1).drop("Heating", axis=1).drop("Description", axis=1).drop("Subdivision", axis=1)

    df = df.join(exteriorData)
    df = df.join(parkingData)
    df = df.join(heatingData)
    df = df.join(descriptionData)
    df = df.join(subdivsion)
    
    test_df = []

    # ChatGPT coded from this part
    def expand_column(df, column):
        expanded = df[column].apply(pd.Series)
        expanded.columns = [f"{column}_{i}" for i in range(expanded.shape[1])]
        return expanded


    for col in ["ExteriorAsInt", "ParkingAsInt", "HeatingAsInt", "SubdivisionAsInt", "DescriptionAsInt"]:
        expanded_cols = expand_column(df, col)
        df = df.drop(col, axis=1).join(expanded_cols)
        test_df = expanded_cols if col == "DescriptionAsInt" else [] # I coded this line

    # To here
    # I used it so I could figure out how to turn the array
    # Inside the row into multiple columns.
    # It was a leaening experiance.

    train_df = df
    '''
    take each house
    get its description
    take the first 30 words
    add it into the train tensor
    repeat until we run out of words
    go to the next house
    repeat
    '''

    for series_name, series in test_df.items():
        train_df = train_df.drop(series_name, axis=1)

    train_data_df = pd.DataFrame()
    counter = 0
    col_count = 0 

    res = []

    for n, _ in train_df.items():
        col_count += 1
        res.append(n)
    if cols:
        return res

    ten_row_completed = False

    words_to_train_jeff_on = 50
    homes_to_train_jeff_on = 20
    for _, i in descriptionData.items():
        j_completed = 0
        for j in range(homes_to_train_jeff_on * amount_homes, min(homes_to_train_jeff_on + (amount_homes * homes_to_train_jeff_on), len(i))):
            dataX = []
            dataY = []
            
            for word in range(len(i[j])-words_to_train_jeff_on):
                if word >= words_to_train_jeff_on:
                    seq_in = i[j][word:word+words_to_train_jeff_on]
                    seq_out = i[j][word+words_to_train_jeff_on]
                else:
                    seq_in = []
                    for _ in range(words_to_train_jeff_on - word):
                        seq_in.append(0)
                    for w in i[j][:word]:
                        seq_in.append(w)

                    seq_out = i[j][word]
                
                dataX.append([k for k in seq_in])
                dataY.append(seq_out)
            
            if idx + words_to_train_jeff_on > len(dataX):
                j_completed += 1       

            for _ in range(idx, min(words_to_train_jeff_on + idx, len(dataX))):
                seq = rand.randint(0, len(dataX))

                temp_df = pd.DataFrame(np.zeros([1, col_count]), columns=[n for n, _ in train_df.items()])

                try:
                    temp = pd.DataFrame([dataX[seq]], columns=[f"des_{l}" for l in range(words_to_train_jeff_on)])
                    temp2 = pd.DataFrame([dataY[seq]], columns=["ans"])
                except Exception:
                    continue

                temp_df = temp_df.join(temp).join(temp2)

                for series_name, series in train_df.items(): 
                    temp_df.loc[j, series_name] = series[j]

                for n, s in temp_df.items():
                    if str(n).startswith("des") or str(n).startswith("ans"):
                        temp_df.loc[j, n] = s[0]

                if train_data_df.empty:
                    train_data_df = pd.DataFrame(np.zeros([1, col_count+words_to_train_jeff_on+1]), columns=[n for n, _ in temp_df.items()])

                for series_name, series in temp_df.items():
                    train_data_df.loc[j + counter, series_name] = series[j]
                counter += 1
            counter -= 1
        if j_completed >= homes_to_train_jeff_on:
            ten_row_completed = True

    '''
        copy train_df's house
        add our data into it. 
        Return it to old data frame 

    '''

    test_data_df = train_data_df["ans"]
    train_data_df = train_data_df.drop("ans", axis=1)

    for i, p in enumerate(train_data_df["Price"]):
        try:
            old_data = train_data_df.loc[i]

            old_data["Price"] = old_data["Price"] / 600
            old_data["SQFT"] = old_data["SQFT"] / 100
            old_data["Acres"] = old_data["Acres"] * 100
        except KeyError:
            continue

    for n, s in train_data_df.items():
        if str(n).startswith("ParkingAsInt_") and int(str(n).replace("ParkingAsInt_", "")) > 20:
            train_data_df = train_data_df.drop(n, axis=1)
        if str(n).startswith("ExteriorAsInt_") and int(str(n).replace("ExteriorAsInt_", "")) > 20:
            train_data_df = train_data_df.drop(n, axis=1)
        if str(n).startswith("HeatingAsInt_") and int(str(n).replace("HeatingAsInt_", "")) > 20:
            train_data_df = train_data_df.drop(n, axis=1)
        if str(n).startswith("SubdivisionAsInt_") and int(str(n).replace("SubdivisionAsInt_", "")) > 6:
            train_data_df = train_data_df.drop(n, axis=1)
    
    train_data_df = train_data_df.fillna(0).apply(pd.to_numeric)
    test_data_df = test_data_df.fillna(0).apply(pd.to_numeric)

    cols_c = 0
    c = []
    for n, _ in train_data_df.items():
        cols_c += 1
        c.append(n)

    train_tensor = torch.tensor(train_data_df.to_numpy(), dtype=torch.long)\
        .unsqueeze(1)

    test_tensor = torch.tensor(test_data_df.to_numpy(), dtype=torch.long)

    xam = 0

    for i, v in enumerate(train_tensor):
        for j, va in enumerate(v):
            for _, value in enumerate(va):
                if value >= xam:
                    xam = value
                    if value >= 2099:
                        train_tensor[i][j] = torch.tensor([0 for _ in range(len(train_tensor[i][j]))])

    return train_tensor, test_tensor, word_to_int, len(wordArr), cols_c, ten_row_completed

if __name__ == "__main__":
    get_tensors()







