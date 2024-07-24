### Setup steps

We tested this on both an Ubuntu 22.04.2 LTS and an Ubuntu 24.04.? LTS 

Run these on the command line one by one:

```
	sudo apt-get update
	sudo apt-get install -y zip build-essential
	sudo apt install tshark
	sudo apt install iw
	wget -L https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
	bash Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b -p ~/miniconda
	rm Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
	echo 'export PATH="~/miniconda/bin:$PATH"' >> ~/.bashrc
	source ~/.bashrc
	conda update conda -y
	conda create -p ~/venv_p310/ python=3.10 -y
	source activate ~/venv_p310/
	pip install numpy scipy matplotlib notebook tqdm torch ultralytics opencv-python PyQt5 cvzone
```

Wireshark's installation needs to be configured according to this (non-root user should be able to use dumpcap): 
https://osqa-ask.wireshark.org/questions/7523/ubuntu-machine-no-interfaces-listed/ 

add an alias for the venv_p310 if you want to make activation easier.

the xcb library is present on both opencv and PyQt5 and they conflict with each other. Do the following to fix this:
https://forum.qt.io/topic/119109/using-pyqt5-with-opencv-python-cv2-causes-error-could-not-load-qt-platform-plugin-xcb-even-though-it-was-found/2 
i.e., delete libqxcb.so from cv2

