# Concrete Crack Image Classification


### 1. Project Description
  * Cracking concrete problem can cause life threatening to people. In order to overcome this problem, a concrete crack image classifiication is developed to identify the concretes with or without crack.
  * The development of concrete crack image classification project here is by using the transfer learning technique. Transfer learning is a technique where a pre-trained neural network model is used as a starting point for a new task instead of training a model from scratch. For my project, at first I set the pretrained feature extractor as non-trainable (freezing) & second time I unfreeze some layers in the feature extractor so that they will receive parameter update.
      
      
### 2. Software used, framework,how to run project
   * Software needed:
     * Visual Studio Code as my IDE. You can get here https://code.visualstudio.com/download
     * Anaconda. https://www.anaconda.com/download

   * Framework:
     * I use Tensorflow and Keras framework to develop this deep learning project, as for Keras it already a part of TensorFlow’s core API.
   
   * How to run project:
     * Download project in the github
     * Download dataset from  link I insert in credits section below.
     * In Visual Studio Code make sure install Python
     * Open Anaconda prompt : "(OPTIONAL) IS FOR GPU INSTALLATION IF YOU NEED FOR CPU THEN IGNORE OPTIONAL"
        * (base) conda create -n "name u want to have" python=3.8
        * (env) conda install -c anaconda ipykernel
        * conda install numpy,conda install pandas,conda install matplotlib (run each of this one by one)
        * (OPTIONAL) conda install -c anaconda cudatoolkit=11.3
        * (OPTIONAL) conda install -c anaconda cudnn=8.2
        * (OPTIONAL) conda install -c nvidia cuda-nvcc
        * conda install git
        * 1 (a) create a folder named TensorFlow inside the tensorflow environment. For example: “C:\Users\< USERNAME >\Anaconda3\envs\tensorflow\TensorFlow”
        * (b) type: cd “C:\Users\<USERNAME>\Anaconda3\envs\tensorflow\TensorFlow” (to change directory to the newly created TensorFlow folder) 
        * (c) type: git clone https://github.com/tensorflow/models.git
        * conda install -c anaconda protobuf
        * 2 (a) type: cd “C:\Users\< USERNAME >\Anaconda3\envs\tensorflow\TensorFlow\models\research” (into TensorFlow\models\research for example)
        * b) type: protoc object_detection/protos/*.proto --python_out=.
        * 3 a) pip install pycocotools-windows
        * b) cp object_detection/packages/tf2/setup.py .
        * c) python -m pip install .
      * Test your installation (RESTART TERMINAL BEFORE TESTING)  
         * Inside C:\Users\< USERNAME > \Anaconda3\envs\tensorflow\TensorFlow\models\research
         * python object_detection/builders/model_builder_tf2_test.py The terminal should show OK if it passes all the tests
      * Open Visual Studio Code, 
         * Go to open new folder, open downloaded file that you download from my repository
         * Make sure downloaded dataset and the concrete_crack.py file in same folder
         * ![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) **ATTENTION!!! : root_path = "Please change the path according to your folder path" DON'T FOLLOW MY PATH IN THE concrete_crack.py FILE SINCE THE PATH IS MY OWN FOLDER PATH**       
         * Then you can run concrete_crack.py file
         * **Troubleshoot: let say you have problem loading the dataset, please check your folder path carefully**
        

 
 
### 3. Results


1. The architecture used in this model is MobileNetV2 which is designed for efficient image clasification to achieve high accuracy on image classification tasks. Implementation of dropout layer in this architecture is to prevent overfitting, that can cause model to perform well on the training data but fails to generalize well to new data.

![model_architecture](https://github.com/dalila28/concreate_crack_image_classification/blob/main/images/architecture.png)

                                                           Model Architecture


2. Model performance


  * Figure 1 showing snapshot of model performance for training under 10 epochs with implementation of early stopping. It manage to complete 10/10 epochs with 99% accuracy for training and validation. 


![model_performance1](https://github.com/dalila28/concreate_crack_image_classification/blob/main/images/model_training_performance1.png)
  
                                                        Figure 1


   * Figure 2 showing snapshot of model performance for continuation of training under 10 fine tune epochs. As you can see from the figure 2 the epochs training is stopped at 18/20 not 20/20 this happen because I applied early stopping during training. "**code line of early stopping: (early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)**". EarlyStopping callback can detect if the model's performance is not improving on new data. If the validation loss does not improve for a certain number of epochs (patience=3) **as you can see for epoch 16 and epoch 18 have similar value of validation loss**, it can be an indication that the model is no longer learning meaningful patterns and further training may not be beneficial. Therefore, training will stop early to avoid overfitting. Training and validation achieved 99% of accuracy and we can say that the model perform optimally as the difference value of training loss and validation loss is not too far.

![model_performance2](https://github.com/dalila28/concreate_crack_image_classification/blob/main/images/model_finetune_p1.png)

                                                       Figure 2


  *   In short, application of early stopping callbacks in the model training can avoid our model to be overfitted.
  
  
3. Tensorboard showing the epoch accuracy of training and validation is 99% of accuracy. 




![tensorboard](https://github.com/dalila28/concreate_crack_image_classification/blob/main/images/tensorboard.png)

                                                      Tensorboard


4. In conclusion, application of transfer learning and early stopping help model to achieve such very high accuracy 99% & not being overfit model.



### 4. Image augmentation


Image augmentation is performed to ensure that model can identify the crack from various angle.



![image](https://github.com/dalila28/concreate_crack_image_classification/blob/main/images/img_augmentation.png)

                                                       Image Augmentation


### 5. Credits
1. The dataset for concrete crack image classificarion is from https://data.mendeley.com/datasets/5y9wdsg2zt/2 Özgenel, Çağlar Fırat (2019), “Concrete Crack Images for Classification”, Mendeley Data, V2, doi: 10.17632/5y9wdsg2zt.2
2. I followed the TensorBoard tutorial provided by TensorFlow, available at https://www.tensorflow.org/tensorboard/get_started, to create visualizations for monitoring and analyzing models.
3. To ensure efficiency in my project, I consistently relied on the comprehensive TensorFlow API documentation at https://www.tensorflow.org/api_docs/python/tf/all_symbols. This documentation served as my go-to resource for exploring the various functions, classes, and modules provided by TensorFlow, enabling me to effectively utilize the powerful TensorFlow framework in my project.

