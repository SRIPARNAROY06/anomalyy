import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, concatenate, Dropout, Lambda, GlobalAveragePooling2D, Dense
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tqdm import tqdm
import matplotlib.pyplot as plt 
from random import randrange
from sklearn.decomposition import PCA
from fastapi import FastAPI, File, UploadFile, Response, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
from skimage.transform import resize
import numpy as np
from io import BytesIO
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Union
import base64
# loaded_model = tf.saved_model.load("my_model")
# print(loaded_model)
class ResponseModel(BaseModel):
    message: str
    image_data: Union[str, None] = None

class AnomalySegmentator(tf.keras.Model):
    def __init__(self, init_layer = 0, end_layer = None):
        super(AnomalySegmentator, self).__init__()
        self.init_layer = init_layer
        self.end_layer = end_layer
        
    def build_autoencoder(self, c0, cd):
        self.autoencoder = Sequential([
            layers.InputLayer((self.map_shape[0]//4, self.map_shape[1]//4, c0)),
            layers.Conv2D((c0 + cd) // 2,(1,1), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.1)),
            layers.Conv2D(2*cd,(1,1), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.1)),
            layers.Conv2D(cd,(1,1), padding='same'),
            layers.Conv2D(2*cd,(1,1), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.1)),
            layers.Conv2D((c0 + cd) // 2,(1,1), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.1)),
            layers.Conv2D(c0,(1,1), padding='same')            
            
        ])
        
    def build(self, input_shape):
        
        self.vgg = VGG19(include_top = False, weights = 'imagenet', input_shape=input_shape[1:])
        self.features_list = [layer.output for layer in self.vgg.layers if 'conv' in layer.name][self.init_layer:self.end_layer]
        
        self.feature_extractor = Model(inputs = self.vgg.input, 
                                       outputs = self.features_list)
        self.feature_extractor.trainable = False   
        
        self.threshold = tf.Variable(0, trainable = False, dtype = tf.float32)

        self.map_shape = self.features_list[0].shape[1:-1]
        
        self.average_pooling = layers.AveragePooling2D(pool_size=(4, 4), strides=(4,4))       
        
        
        
        self.c0 = sum([feature.shape[-1] for feature in self.features_list])        
        self.cd = 40        
        self.build_autoencoder(self.c0, self.cd)
        
          
    def __call__(self, inputs):
        features = self.feature_extractor(inputs)
        resized_features = [tf.image.resize(feature, self.map_shape) for feature in features]
        resized_features = tf.concat(resized_features, axis = -1)
        
        resized_features = self.average_pooling(resized_features)

        autoencoder_output = self.autoencoder(resized_features)
        return tf.reduce_sum((autoencoder_output - resized_features)**2, axis = -1)
        
    def reconstruction_loss(self):
        @tf.function
        def _loss(y_true, y_pred):
            loss = tf.reduce_mean(y_pred, axis = (1,2)) / (tf.cast(tf.shape(y_pred)[0], tf.float32) * self.c0)
            return loss
                    
        return _loss

    def compute_threshold(self, data_loader, fpr = 0.05):
        error = []
        for i in tqdm(range(len(data_loader))):
            x, y = data_loader[i]
            error.append(self(x))
        error = np.concatenate(error)
        threshold = np.percentile(error, 100 - fpr)
        self.threshold = tf.Variable(threshold, trainable = False, dtype = tf.float32)
        return error
    
    def compute_pca(self, data_loader):
        extraction_per_sample = 20
        
        extractions = []        
        for i in tqdm(range(len(data_loader))):
            x, _ = data_loader[i]     
            
            features = self.feature_extractor(x)
            resized_features = [tf.image.resize(feature, self.map_shape) for feature in features]
            resized_features = tf.concat(resized_features, axis = -1)

            resized_features = self.average_pooling(resized_features)
            
            for feature in resized_features:
                
                for _ in range(extraction_per_sample):                    
                
                    row, col = randrange(feature.shape[0]), randrange(feature.shape[1])
                    extraction = feature[row, col]
                    extractions.append(extraction)
            
        extractions = np.array(extractions)
        print(f"Extractions Shape: {extractions.shape}")
        pca = PCA(0.9, svd_solver = "full")
        pca.fit(extractions)
        self.cd = pca.n_components_
        self.build_autoencoder(self.c0, self.cd)
        print(f"Components with explainable variance 0.9 -> {self.cd}")
  
# loaded_model = tf.saved_model.load("my_model")
# print(loaded_model) 
def apply_mask(image, mask, transparency=0.4):

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    # Convert the mask to a 3-channel image if it's not already
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Normalize the mask to be in range [0, 1]
   # mask_normalized = mask.astype(float) / 255.0
    mask_converted = mask.astype(np.float32) # Convert mask to float32 (same as image)
    blended_image = cv2.addWeighted(image.astype(np.float32), 1 - transparency, mask_converted, transparency, 0)

    # Blend the image and the mask using the transparency value
    #blended_image = cv2.addWeighted(image, 1 - transparency, mask, transparency, 0)
    blended_image = (blended_image).astype(np.uint8)
    return blended_image
INPUT_SIZE = (224,224)      
loaded_model = AnomalySegmentator()
loaded_model.compile(Adam(1e-4), loss = loaded_model.reconstruction_loss())
loaded_model.build((None, *INPUT_SIZE,3))
loaded_model.load_weights('anomaly-segmentation-model.h5')
# print("model loaded")
app = FastAPI()
@app.middleware("http")
async def ignore_favicon(request: Request, call_next):
    if request.url.path == "C:/Users/SRIPARNA ROY/Downloads/archive (5)/images/favicon.ico":
        return Response(status_code=404)
    return await call_next(request)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
@app.get("/")
async def root():
    return {"message": "API"}
@app.post("/predict", response_model=ResponseModel)
async def predict(file: UploadFile = File(...)):
  contents = await file.read()
  nparr = np.frombuffer(contents, np.uint8)
  x = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  # x = cv2.imread(contents)
  x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
  x_resized = cv2.resize(x, (224, 224))
  x_resized = np.expand_dims(x_resized, axis=0) / 255.

  # Anomaly prediction using the model
  heatmap = loaded_model(x_resized)
  heatmap = np.squeeze(heatmap, axis=0)
  heatmap_resized = resize(heatmap, x.shape[:-1], anti_aliasing=True)

  # Thresholding for anomaly segmentation mask
  mask = np.where(heatmap_resized > loaded_model.threshold, 1, 0)

  # Generate segmented image
  segmented_image = x * np.stack([mask]*3, axis=-1)
  prediction = np.any(heatmap_resized > loaded_model.threshold)

  # Apply mask with transparency and save image
 # blended_image = apply_mask(segmented_image, mask)
  #cv2.imwrite('seg_img.jpg', blended_image)
# Create subplots for original image and mask
  fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot the original image
  ax1.imshow(x[:, :, ::-1])  # Convert BGR to RGB for matplotlib
  ax1.set_title('Original Image')
  ax1.axis('off')

    # Plot the mask (adjust colormap and thresholding if needed)
  ax2.imshow(heatmap_resized, cmap='hot')  # Use a heatmap colormap
  ax2.set_title('Anomaly Mask')
  ax2.axis('off')

    # Tight layout to avoid overlapping titles
  plt.tight_layout()

    # Save the plot with original image and mask
  plt.savefig('anomaly_segmentation.jpg')
  plt.close(fig)
  image = cv2.imread('anomaly_segmentation.jpg')
  _, buffer = cv2.imencode('.jpg', image)
  img_str = base64.b64encode(buffer).decode("utf-8")
  prediction = bool(prediction)
    # Return the response with the saved plot image
  response = {
        "Anomaly Detected": prediction,
        "image_data": img_str
    }
  return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)