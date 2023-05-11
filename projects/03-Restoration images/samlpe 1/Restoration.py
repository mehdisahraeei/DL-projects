#clone git
!git clone https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git


#downloading and coping other libraries in main folder
%cd /content/Bringing-Old-Photos-Back-to-Life/Face_Enhancement/models/networks
!git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
!cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
%cd ../../../

%cd /content/Bringing-Old-Photos-Back-to-Life/Global/detection_models
!git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
!cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
%cd ../../

%cd /content/Bringing-Old-Photos-Back-to-Life/Face_Detection
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
%cd ../


%cd /content/Bringing-Old-Photos-Back-to-Life/Face_Enhancement
!wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip
!unzip face_checkpoints.zip
%cd ../


%cd /content/Bringing-Old-Photos-Back-to-Life/Global
!wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
!unzip global_checkpoints.zip
%cd ../



#installing requirements.txt
!pip install -r /content/Bringing-Old-Photos-Back-to-Life/requirements.txt


#final , run
!python run.py --input_folder /content/Bringing-Old-Photos-Back-to-Life/test_images/old --output_folder /content/Bringing-Old-Photos-Back-to-Life/output --GPU 0



#showing
from IPython.display import Image

#old image
Image('/content/Bringing-Old-Photos-Back-to-Life/test_images/old/h.png')

#restored image
Image('/content/Bringing-Old-Photos-Back-to-Life/output/final_output/h.png')
