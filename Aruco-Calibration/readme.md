tumu icin opencv versiyonu 4.7 den buyuk olmali

detect_aruco_image.py - python3 detect_aruco_image.py -t DICT_6X6_250 -i 1.jpg
detect_aruco_video.py - python3 detect_aruco_video.py -t DICT_6X6_250
aruco_generate.py - python3 aruco_generate.py --type DICT_6X6_250 --output /output/path --id 10

detect_aruco_video.py kamera sourcesi ayarlanmali ek olarak. 

https://github.com/KhairulIzwan/ArUco-markers-with-OpenCV-and-Python reposundan yardim aldim.

1 metre x 1 metre 3 arucomarkerdan olusan kucuk bir chessboard yaptim. Onun uzerinden cok hizli bir sekilde kalibre ediyor ama chessboarddan uzaklastikca 5cm ye kadar error gordum. CalPnP nin daha hizli hali olarak yapabildim. Internette 2 ya da 3 tane ornek buldum hepside detect edilen Aruco Markerlarin gercek dunyadaki kordinatlarini kullanarak homography matrix uretiyor. Bende tam olarak oyle yaptim. Marker sayisini arttirip error ne kadar dusuyor tekrar test edecegim (yazicinin kartusu az kalmisti 3 tane marker tek bastirabildim bastirdigim diger markerlar silik olunca detect edemedi).




