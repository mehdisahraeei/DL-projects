!git clone https://github.com/ultralytics/yolov5

%cd yolov5

!pip install -r requirements.txt



from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)


!pip install kaggle

%ls
#upload your json file in yolov path

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/yolov5" #yolov path


!chmod 600 /content/yolov5/kaggle.json
!kaggle datasets download -d ahmedhaytham/car-detection --force


!unzip /content/yolov5/car-detection.zip


!python train.py --img 415 --batch 16 --epochs 30 --data /content/yolov5/data.yaml --weights yolov5x.pt --cache #--cache showing details


!python export.py --weights /content/yolov5/runs/train/exp/weights/best.pt --include tflite --img 416

%cd /content/yolov5/runs/train/exp

!zip -r log.zip /content/yolov5/runs/train/exp/weights/
