# Predict Speed from Video
### Datasets


#####Training 
*  17 minutes of .mp4 dashcam footage at 20 fps, processed to 20400 labelled frames 

#####Test
* 8 minutes 59 seconds of dashcam footage at 20 fps

### Data Augmentation

Dropout, brightness and flipping of training data

## Model

5 convolutional layer groups and 1 fully connected layer group pretained on weights from the sports-1M dataset

```_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1 (Conv3D)               (None, 16, 112, 112, 64)  5248      
_________________________________________________________________
pool1 (MaxPooling3D)         (None, 16, 56, 56, 64)    0         
_________________________________________________________________
conv2 (Conv3D)               (None, 16, 56, 56, 128)   221312    
_________________________________________________________________
pool2 (MaxPooling3D)         (None, 8, 28, 28, 128)    0         
_________________________________________________________________
conv3a (Conv3D)              (None, 8, 28, 28, 256)    884992    
_________________________________________________________________
conv3b (Conv3D)              (None, 8, 28, 28, 256)    1769728   
_________________________________________________________________
pool3 (MaxPooling3D)         (None, 4, 14, 14, 256)    0         
_________________________________________________________________
conv4a (Conv3D)              (None, 4, 14, 14, 512)    3539456   
_________________________________________________________________
conv4b (Conv3D)              (None, 4, 14, 14, 512)    7078400   
_________________________________________________________________
pool4 (MaxPooling3D)         (None, 2, 7, 7, 512)      0         
_________________________________________________________________
conv5a (Conv3D)              (None, 2, 7, 7, 512)      7078400   
_________________________________________________________________
conv5b (Conv3D)              (None, 2, 7, 7, 512)      7078400   
_________________________________________________________________
zeropad5 (ZeroPadding3D)     (None, 2, 8, 8, 512)      0         
_________________________________________________________________
pool5 (MaxPooling3D)         (None, 1, 4, 4, 512)      0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0         
_________________________________________________________________
fc6 (Dense)                  (None, 4096)              33558528  
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
fc7 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
dropout_2 (Dropout)          (None, 4096)              0         
_________________________________________________________________
fc8 (Dense)                  (None, 1)                 4097      
=================================================================
Total params: 77,999,873
Trainable params: 77,999,873
Non-trainable params: 0
_________________________________________________________________
```
## Training

```sgd = SGD(lr=1e-5, decay=0.001, momentum=0.9)```

## Challenges

This was a brute force approach to learning how to setup an ML development environment, research some models and apply Keras to experiment with a small dataset. I've got a huge amount to learn but have enjoy the challenge.
It was easy enough to get the MSE on the validation set below 3 but this created an over trained model. I experimented with dropout, early stopping, and L1 and L2 regularization to overcome over training and did the best I could with the data available.
## Result

* https://github.com/tonypls/simpleSpeed/test_pred.txt
* https://github.com/tonypls/simpleSpeed/test_out.mp4

## Future Work

* Improve accuracy 
* Implement an LSTM to give the model some "memory"
* Apply some smoothing to the result, limit the rate of change
* Remove some noise but identifying and exluding other moving vehicles
 

## References

* https://github.com/experiencor/speed-prediction