# Concrete Crack Image Classification


### 1. Project Description
  * Cracking concrete problem can cause life threatening to people. In order to overcome this problem, a concrete crack image classifiication is developed to identify the concretes with or without crack.
  * In the development of this project, running the 
      
      
    

### 2. Software used, framework,how to run project
   * Software needed:
     * Visual Studio Code as my IDE. You can get here https://code.visualstudio.com/download
     * Anaconda. https://www.anaconda.com/download

   * Framework:
     * I use Tensorflow and Keras framework to develop this deep learning project, as for Keras it already a part of TensorFlow’s core API.
   
   * How to run project:
     * Download project in the github
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
         * Go to open new folder, open downloaded file that you download from my repository, then you can run concrete_crack.py file
        

 
 
### 3. Results


1. The architecture used in this model is MobileNetV2 which is designed for efficient image clasification to achieve high accuracy on image classification tasks. Implementation of dropout layer in this architecture is to prevent overfitting, which cause model to perform well on the training data but fails to generalize well to new data.

![model_architecture](https://github.com/dalila28/concreate_crack_image_classification/blob/main/images/architecture.png)

                                                           Model Architecture


2. Model performance


  * Figure 1 showing snapshot of model performance for training under 10 epochs 


![model_performance1](https://github.com/dalila28/concreate_crack_image_classification/blob/main/images/model_training_performance1.png)
  
                                                                Figure 1


   * Figure 2 showing snapshot of model performance for continuation of training under 10 fine tune epochs. As you can see from the figure 2 the epochs training is stopped at 18/20 not 20/20 this happen because I applied early stopping during training. "**code line of early stopping: (early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)**". EarlyStopping callback can detect if the model's performance is not improving on new data. If the validation loss does not improve for a certain number of epochs (patience=3), it can be an indication that the model is no longer learning meaningful patterns and further training may not be beneficial. Therefore, training can be stopped early to avoid overfitting.


![model_performance2](https://github.com/dalila28/concreate_crack_image_classification/blob/main/images/model_finetune_p1.png)

                                                                Figure 2

3. Tensorboard snapshot showing graph of MSE

![tensorboard]


4. Figure below showing the matplotlib graph comparison between actual & predicted result of covid-19 case in Malaysia based on my deep learning project. From the graph we can see that the predicted line is following the curve of actual line which as for my observation I can say that the result is good eventhough it not following correctly the spike of curve. If we want to improve the result, I think we can increase number of epochs so model has more opportunities to learn from the data and adjust its parameters to improve performance.
![actual_vs_predicted](https://github.com/dalila28/covid19-case-prediction/blob/main/images/actual_vs_predicted.png)



