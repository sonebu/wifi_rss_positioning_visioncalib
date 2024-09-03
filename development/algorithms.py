from sklearn.cluster import KMeans 
import math, torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

#from auxiliary import load_data, prepare_data_loaders# auxiliary.py'de tanımlı olan load_data, prepare_data_loaders ve MLP kullanılacak

#################################################################################################################################################################
### Nearest Neighbour
###

## inference function (predicts loc based on rss input)
def RssPositioningAlgo_NearestNeighbour(rss_values, db): # db is a Tuple of the form (<rssi array>,<loc array>)
    min_distance = float('inf')
    best_match_xy = None

    # check if the length of data lists in db matches those submitted for prediction
    if(len(db[0][0]) != len(rss_values)):
        raise ValueError(f'Length of the submitted RSSI array ({len(rss_values)}) for prediction does not match the length of that in the database ({len(db[0][0])})')

    # exhaustive search within db, iterate over rssi elements and extract corresponding locs
    for fp_idx, fp in enumerate(db[0]):
        fingerprint  = fp
        ref_location = db[1][fp_idx]
        distance     = math.dist(rss_values, fingerprint)
        if distance < min_distance:
            min_distance = distance
            best_match_xy   = ref_location
    return best_match_xy # best_match_xy is an array of size 1x2, for loc_x and loc_y respectively

## inference function (predicts loc based on rss input), uses interpolation on top of nearest neighbour
def RssPositioningAlgo_NearestNeighbour_Interpolation(rss_values, db, interp_k=3): # k=3 default olarak verildi
    distances = []    
    for fp_idx, fp in enumerate(db[0]): # data_dictionary'deki her bir eleman için
        fingerprint  = fp
        ref_location = db[1][fp_idx]
        distance = math.dist(rss_values, fingerprint) # rss_values ve fingerprint arasındaki mesafeyi hesapla
        distances.append((distance, ref_location)) # mesafe ve ref_location'ı distances listesine ekle
    
    distances.sort(key=lambda x: x[0]) # distances listesini mesafeye göre sırala
    nearest_neighbors = distances[:interp_k] # en yakın interp_k elemanı nearest_neighbors listesine ekle
   
    weighted_sum_x = 0 # x değerlerinin ağırlıklı toplamı
    weighted_sum_y = 0 # y değerlerinin ağırlıklı toplamı
    total_weight = 0 # toplam ağırlık
    
    for distance, position in nearest_neighbors: # nearest_neighbors listesindeki her bir eleman için
        weight = 1 / (distance**2) if distance != 0 else 1e10   # ağırlık hesapla (1/r^2 modeli)
        x, y = position # position'ı x ve y'ye ayır
        weighted_sum_x += weight * x # x değerlerinin ağırlıklı toplamını
        weighted_sum_y += weight * y    # y değerlerinin ağırlıklı toplamını 
        total_weight += weight # toplam ağırlığı güncelle

    estimated_x = weighted_sum_x / total_weight # tahmini x değeri
    estimated_y = weighted_sum_y / total_weight # tahmini y değeri
    
    return (estimated_x, estimated_y) # tahmini x ve y değerlerini döndür

## "training" function (builds database using Kmeans, based on provided training set xt-yt)
def RssPositioningAlgo_NearestNeighbour_GetKmeansDb(xt, yt, num_clusters, verbose=False):
    kmeans_rss = KMeans(n_clusters = num_clusters, random_state=0, n_init="auto").fit(yt) 
    train_cluster_ids = kmeans_rss.labels_
    if(verbose):
        print("Cluster IDs of each train set element after Kmeans:\n")
        print(train_cluster_ids)
        print("-"*20)
    db_kmeans_locs = kmeans_rss.cluster_centers_
    if(verbose):
        print("Cluster centers (location) of each cluster center (virtual point), {x,y}:\n")
        print(db_kmeans_locs)
        print("-"*20)
    train_cluster_rss_means = np.zeros((num_clusters,3))
    train_cluster_ctrs      = np.zeros((num_clusters,1)) # to be used for averaging
    for db_idx, cluster_idx in enumerate(train_cluster_ids):
        train_cluster_rss_means[cluster_idx] += xt[db_idx].numpy()
        train_cluster_ctrs[cluster_idx]      += 1
    db_kmeans_RSSIs = train_cluster_rss_means / train_cluster_ctrs
    db_kmeans = (db_kmeans_RSSIs, db_kmeans_locs)
    return db_kmeans


#4- MLP-based supervised model


def mlp_based_localization(train_loader,tensor_x_train,number_of_training_iters, batch_size):

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
        
        if (i+1) % 20 == 0: # her 20 epoch'ta
            print(f'Epoch [{i + 1}/{number_of_training_iters}] running accumulative loss across all batches: {running_loss:.3f}')

    
    predicted_locations_trainset = model(tensor_x_train) # eğitim seti için tahmin edilen konumları al
    #print("Predicted locations on training set:", predicted_locations_trainset)

    
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
