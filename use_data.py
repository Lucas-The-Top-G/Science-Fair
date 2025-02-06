import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from get import get_tensors
from model import DescriptionWriter

def name_goes_here():
    path = "models/FINAL_AI_BUILD2.pth"

    idx = 0
    amount_homes = 0

    best_model = None

    batch_size = 45
    epochs = 125

    input_size = None
    hidden_size = 200
    output_size = 2100
    num_layers = 4
    
    try:
        saved_data = torch.load(path, weights_only=True)
        
        best_model = saved_data[0]
        word_to_int = saved_data[1]
        input_size = saved_data[2]
        hidden_size = saved_data[3]
        output_size = saved_data[4]
        idx = saved_data[5]
        amount_homes = saved_data[6]
        num_layers = saved_data[7]
    except FileNotFoundError:
        pass

    if idx > 300:
        idx = 0
        amount_homes += 1

    X, Y, word_to_int, _, Z, UP = get_tensors(idx=idx, amount_homes=amount_homes)
    idx += 15

    if input_size != Z:
        input_size = Z

    if UP:
        amount_homes += 1
        idx = 0

    model = DescriptionWriter(
                            input_size=input_size, 
                            hidden_size=hidden_size, 
                            output_size=output_size,
                            num_layers=num_layers,
                            )
    
    if best_model != None:
        model.load_state_dict(best_model)

    criteria = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.975)

    loader = data.DataLoader(data.TensorDataset(X, Y), shuffle=True, batch_size=batch_size)

    best_model = None
    best_loss = np.inf

    losses_array = []

    try:
        print(f"Starting Training, Input: {input_size}, Hidden: {hidden_size}, Output Size :{output_size}, Num Layers: {num_layers}")
        for epoch in range(epochs):
            model.train()

            for X_batch, Y_batch in loader:

                y_predict = model(X_batch)

                loss = criteria(y_predict, Y_batch)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            model.eval()
            loss = 0

            with torch.no_grad():
                for X_batch, Y_batch in loader:
                    y_predict = model(X_batch)

                    loss += criteria(y_predict, Y_batch)

                if loss < best_loss:
                    best_loss = loss
                    best_model = model.state_dict()

                losses_array.append(int(loss))
                
                print(f"{epoch} / {epochs} Epochs: lr: {optimizer.param_groups[0]['lr']} Cross-entropy: {loss} ")
                if loss < 1.1e-4:
                    break
            scheduler.step()
    except KeyboardInterrupt:
        pass
    print(losses_array)
    torch.save([best_model, word_to_int, input_size, hidden_size, output_size, idx, amount_homes, num_layers], path)  

    plt.plot([i for i in range(len(losses_array))], losses_array)
    plt.show()

    return True
# if loss <= 0.1 and epoch > 10:
#     print(f"{epoch} / {epochs} Epochs: This epoch got to a loss of below 0.09, moving to next data")
#     break

# elif loss+0.05 >= l.avg and loss-0.05 <= l.avg and loss >= 20:
#     can_run = 0
    
#     for item in l.loss_array:
#         if loss+0.1 >= item and loss-0.1 <= item:
#             can_run += 1
    
#     if can_run >= l.length-2:
#         print(f"{epoch} / {epochs} Epochs: This Epoch is not moving enough, moving to next data")
    
#         break

# l.add_to_array(loss, epoch)

class losses():
    def __init__(self, length_of_data):
        self.loss_array = [np.inf for _ in range(length_of_data)]
        self.avg = 0
        self.length = length_of_data

    def get_average(self):
        running = 0
        total = 0
        for i in self.loss_array:
            total += i
            running += 1
        self.avg = total / running

    def add_to_array(self, item, epoch):
        self.loss_array[epoch % self.length] = item
        self.get_average()
