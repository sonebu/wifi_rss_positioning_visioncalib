 
import math
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from auxiliary import data_dictionary  # auxiliary.py'de tanımlı olan data_dictionary kullanılacak
from auxiliary import load_data, prepare_data_loaders# auxiliary.py'de tanımlı olan load_data, prepare_data_loaders ve MLP kullanılacak

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
            
    x, y = map(float, best_match.split('_'))
    return [x,y]



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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(3, 16)
        self.hidden_layer1 = nn.Linear(16, 32)
        self.hidden_layer2 = nn.Linear(32, 20)
        self.output_layer = nn.Linear(20, 2)
        self.activation_fcn = nn.ReLU()

    def forward(self, x):
        x = self.activation_fcn(self.input_layer(x))
        x = self.activation_fcn(self.hidden_layer1(x))
        x = self.activation_fcn(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x



test_rss_values = [-34, -60, -54]


# 1- Test Nearest Neighbor with One Point
#print("\nRunning Nearest Neighbor with One Point...")
#nn_single_point_result = RssPositioningAlgo_NearestNeighbour(test_rss_values, data_dictionary)
#print("Nearest Neighbor (One Point) Result:", nn_single_point_result)

# 3- Test Nearest Neighbor with Multiple Points + Interpolation
#print("\nRunning Nearest Neighbor with Multiple Points + Interpolation...")
#nn_interpolation_result = RssPositioningAlgo_Interpolation(test_rss_values, data_dictionary, k=3)
#print("Nearest Neighbor with Interpolation Result:", nn_interpolation_result)
