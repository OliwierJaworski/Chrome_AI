import os
from ultralytics import YOLO
import torch

test_image_dir = "results/before"
test_image_name = "sprite_multiple.jpg"
result_image_dir = "results/after"

pretrained = True 
save_export_file = False #inverted logic for some reason
epochs = 20

pretrained_path = "yolo11n.pt" #which model will be used if pretrained is True
mode_save_name = "trained.pt" #where the model will be saved 

def main():
    if __debug__ : 
        print("Program execution started!\n")   
        print(os.getcwd())
    model = model_load()
    #training_results, validation_results = train_model(model)
    save_export(model=model)  # Uncomment to enable export
    model_test(model=model)

def model_load():
    if not pretrained:
        model = YOLO("yolo11n.pt")
    else:
        model = YOLO(model=pretrained_path)
    return model

# If training is enabled 
def train_model(model):
    if not pretrained:
        #training_results = model.train(data="Fles_dataset.yaml", epochs=epochs, imgsz=640)
        training_results = model.train(data="datainfo.yaml", epochs=epochs, imgsz=640)
        validation_results = model.val()  # Separate validation results
        return training_results, validation_results
    
# If export is enabled
def save_export(model):
    if save_export_file:
        model.save(mode_save_name)
        model.export(format="onnx", opset=11)

# Will test the model on image provided 
def model_test(model):
    result = model(test_image_dir + "/" + test_image_name)
    result[0].save(result_image_dir + "/" + test_image_name)

if __name__ == "__main__":
    main()
