 
import math
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from auxiliary import data_dictionary  # auxiliary.py'de tanımlı olan data_dictionary kullanılacak
from auxiliary import load_data, prepare_data_loaders, MLP # auxiliary.py'de tanımlı olan load_data, prepare_data_loaders ve MLP kullanılacak

#1- nearest neighbour az pt            ||    
#2- nearest neighbour çok pt           || --> bu metodlar için şu commit ID'de implementation var: f3a2db49692c0928cd671d593489ffb04ea6e9ba     

def RssPositioningAlgo_NearestNeighbour(rss_values, data_dictionary):
    min_distance = float('inf')
    best_match = None

    for position, fingerprint in data_dictionary.items():
        distance = math.dist(rss_values, fingerprint)
        if distance < min_distance:
            min_distance = distance
            best_match = position
            
    return best_match



#3- nearest neighbour çok pt + interp  --> bunun implementation'ı github'da yok şu anda (1/r^2 model kullanılmıştı)

def RssPositioningAlgo_Interpolation(rss_values, data_dictionary, k=3): # k=3 default olarak verildi
    distances = []
    
    for position, fingerprint in data_dictionary.items(): # data_dictionary'deki her bir eleman için
        distance = math.dist(rss_values, fingerprint) # rss_values ve fingerprint arasındaki mesafeyi hesapla
        distances.append((distance, position)) # mesafe ve position'ı distances listesine ekle
    
    
    distances.sort(key=lambda x: x[0]) # distances listesini mesafeye göre sırala
    nearest_neighbors = distances[:k] # en yakın k elemanı nearest_neighbors listesine ekle
    
   
    weighted_sum_x = 0 # x değerlerinin ağırlıklı toplamı
    weighted_sum_y = 0 # y değerlerinin ağırlıklı toplamı
    total_weight = 0 # toplam ağırlık
    
    for distance, position in nearest_neighbors: # nearest_neighbors listesindeki her bir eleman için
        weight = 1 / (distance**2) if distance != 0 else 1e10   # ağırlık hesapla
        x, y = map(float, position.split('_')) # position'ı x ve y'ye ayır
        weighted_sum_x += weight * x # x değerlerinin ağırlıklı toplamını
        weighted_sum_y += weight * y    # y değerlerinin ağırlıklı toplamını 
        total_weight += weight # toplam ağırlığı güncelle

    estimated_x = weighted_sum_x / total_weight # tahmini x değeri
    estimated_y = weighted_sum_y / total_weight # tahmini y değeri
    
    return (estimated_x, estimated_y) # tahmini x ve y değerlerini döndür



#4- MLP-based supervised model



def mlp_based_localization(json_file, number_of_training_iters=401, batch_size=1, train_test_split=0.5): # number_of_training_iters=401, batch_size=1, train_test_split=0.5 default olarak verildi
  
    inp_rss_vals, gt_locations = load_data(json_file)  # load_data fonksiyonunu kullanarak veriyi yükle
    train_loader, test_loader, tensor_x_train, tensor_y_train, tensor_x_test, tensor_y_test = prepare_data_loaders(
        inp_rss_vals, gt_locations, batch_size, train_test_split)  # prepare_data_loaders fonksiyonunu kullanarak veriyi hazırla

 
    model = MLP() # MLP modelini oluştur
    model.train() # modeli eğitim moduna al

  
    criterion = nn.MSELoss(reduction='mean') # loss fonksiyonu olarak Mean Squared Error kullan
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # optimizer olarak Adam kullan

    
    for i in range(number_of_training_iters): # number_of_training_iters kadar eğitim yap
        running_loss = 0.0 # başlangıçta loss değeri 0
        for inputs, labels in train_loader: # train_loader'dan her bir batch için
            optimizer.zero_grad()   
            outputs = model(inputs)     
            loss = criterion(outputs, labels)   
            loss.backward()            
            optimizer.step()         
            running_loss += loss.item() 

        if i+1 % 1 == 0: # her 20 epoch'ta
            print(f'Epoch [{i + 1}/{number_of_training_iters}] running accumulative loss across all batches: {running_loss:.3f}')

    
    predicted_locations_trainset = model(tensor_x_train) # eğitim seti için tahmin edilen konumları al
    #print("Predicted locations on training set:", predicted_locations_trainset)

    
    predicted_locations_test = model(tensor_x_test) # test seti için tahmin edilen konumları al
    #print("Predicted locations on test set:", predicted_locations_test)

    return model


#5- nearest neighbour çok pt + MLP ...


def combined_fingerprint_mlp_localization(json_file, k=3, number_of_training_iters=401, batch_size=32, train_test_split=0.8): # k=3, number_of_training_iters=401, batch_size=32, train_test_split=0.8 default olarak verildi
    # Load and prepare data
    inp_rss_vals, gt_locations = load_data(json_file)
    train_loader, test_loader, tensor_x_train, tensor_y_train, tensor_x_test, tensor_y_test = prepare_data_loaders(
        inp_rss_vals, gt_locations, batch_size, train_test_split)


    # Step 1: Select k nearest neighbors for each test point 
    selected_train_indices = []
    selected_train_inputs = []
    selected_train_outputs = []
    
    for test_point in tensor_x_test:
        distances = torch.norm(tensor_x_train - test_point, dim=1) 
        nearest_indices = torch.topk(distances, k=k, largest=False).indices  # Find k nearest neighbors
        selected_train_indices.append(nearest_indices)
        
        # Collecting the corresponding RSS and locations of nearest neighbors
        selected_train_inputs.extend(tensor_x_train[nearest_indices])
        selected_train_outputs.extend(tensor_y_train[nearest_indices])

    selected_train_inputs = torch.stack(selected_train_inputs)
    selected_train_outputs = torch.stack(selected_train_outputs)

    # Step 2: Train an MLP on the selected training set
    selected_train_dataset = torch.utils.data.TensorDataset(selected_train_inputs, selected_train_outputs)
    selected_train_loader = torch.utils.data.DataLoader(selected_train_dataset, batch_size=batch_size, shuffle=True)
    
    model = MLP()
    model.train()

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(number_of_training_iters):
        running_loss = 0.0
        for inputs, labels in selected_train_loader:
            optimizer.zero_grad()       # Zero the gradients
            outputs = model(inputs)     # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()             # Backward pass
            optimizer.step()            # Update weights
            running_loss += loss.item() # Accumulate loss

        if i % 1 == 0:
            print(f'Epoch [{i + 1}/{number_of_training_iters}] running accumulative loss across all batches: {running_loss:.3f}')

    # Step 3: Evaluate the model on the test set 
    model.eval()
    predicted_locations_test = model(tensor_x_test)
    #print("Predicted locations using combined Fingerprint and MLP:", predicted_locations_test)

    return model


test_rss_values = [-34, -60, -54]

#####        EXAMPLE USAGES     #####


# 1- Test Nearest Neighbor with One Point
print("\nRunning Nearest Neighbor with One Point...")
nn_single_point_result = RssPositioningAlgo_NearestNeighbour(test_rss_values, data_dictionary)
print("Nearest Neighbor (One Point) Result:", nn_single_point_result)

# 3- Test Nearest Neighbor with Multiple Points + Interpolation
print("\nRunning Nearest Neighbor with Multiple Points + Interpolation...")
nn_interpolation_result = RssPositioningAlgo_Interpolation(test_rss_values, data_dictionary, k=3)
print("Nearest Neighbor with Interpolation Result:", nn_interpolation_result)

# 4- Test MLP-based Localization
model = MLP()
print("\nRunning MLP-based Localization...")
model = mlp_based_localization('data.json', number_of_training_iters=10, batch_size=32, train_test_split=0.8)
model.eval()
output = model(torch.tensor(test_rss_values).float())
print("MLP-based localization results: ", output.tolist())

# 5- Test Nearest Neighbor with Multiple Points + MLP
model1 = MLP()
print("\nRunning Nearest Neighbor with Multiple Points + MLP...")
model1 = combined_fingerprint_mlp_localization('data.json', k=3, number_of_training_iters=10, batch_size=32, train_test_split=0.8)
model.eval()
output = model1(torch.tensor(test_rss_values).float())
print("est Nearest Neighbor with Multiple Points + MLP results: ", output.tolist())