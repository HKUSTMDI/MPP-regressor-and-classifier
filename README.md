# MPP-regressor-and-classifier
In this project, we trained a model to regress the mpp (Microns Per Pixel
) value of patches on HE-stained breast tissue WSI. According to relevant research, visual foundation models trained on a specific mpp, such as UNI, will perform better on test data with a similar mpp to their training data. Below are the patches of breast WSI captured under three different mpps we selected. It can be seen that the image representations of different mpps are quite different.
<img width="712" alt="截屏2025-04-25 下午1 16 58" src="https://github.com/user-attachments/assets/1bbbdc57-6211-419e-889f-1b56a5308c79" />
We all know that pathological slice WSI files are saved by level, and the resolution of different layer images is different. The higher the layer, the lower the magnification, the larger the mpp, and the magnification is 2. In our training, we selected patches with mpp values ​​corresponding to WSI at magnifications of x40, x35, x30, x25, x20, x15, x10, and x5 to train the regressor, and the backbone of the model is ResNet50.

We set the ratio of training set, validation set, and test set to 8:1:1, and used regression evaluation indicators on the test set to verify the effect, and achieved good results.
![20250425-133540](https://github.com/user-attachments/assets/8d91fb15-4f00-4204-bc70-ec85c836fbad)
MAE  : 0.0811
MSE  : 0.0199
RMSE : 0.1412
R²   : 0.9601


