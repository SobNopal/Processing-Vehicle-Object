from ultralytics import YOLO

model=YOLO('yolov8n.pt')
datasetPath="D:/NAUFAL/File Naufal/Semester 6/Pengolahan Citra Digital/Naufal/data.yaml"
modelPath="D:/NAUFAL/File Naufal/Semester 6/Pengolahan Citra Digital/Naufal/model"
model.train(data=datasetPath, task='detect', project=modelPath, imgsz=640, save=True, half=False, patience=0, epochs=200, device='cpu')
