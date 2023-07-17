# an end to end application of classification of images using streamlit


<img width="841" alt="Screenshot 2023-07-17 at 6 59 49 PM" src="https://github.com/cs25-esc/assignment1/assets/68850280/ee47f23a-163f-47fc-8fbd-b5832cc2863e">


step1 - upload the image
step2 - click the button for classification


# method

the trained deep learning model has been converted into a .h5 file so that we can use the model in the streamlit application


# result

![image](https://github.com/cs25-esc/assignment1/assets/68850280/29eec6d6-be5b-4e85-8d1a-37d7db62d57f)






# assignment1
classification CNN model

The below code snippet helps to unzip the dataset

import zipfile
zip_ref = zipfile.ZipFile('/content/intel-image-classification.zip' , 'r')
zip_ref.extractall('/content')
zip_ref.close()


The below code snippet the train_dataset variable will contain a TensorFlow tf.data.Dataset object that represents the image dataset, ready for training of the model. Each batch in the dataset will consist of 32 images with a size of 224x224 pixels.

train_dataset = keras.utils.image_dataset_from_directory(
    directory = '/content/seg_train/seg_train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (224 , 224)
)

The below code snippet: creates a sequential model in Keras with multiple layers. 
After creating the model, you can compile it and proceed with training using appropriate optimizer, loss function, and metrics.

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(6)
])

below code: helps to compile and evaluate the model

model.compile(optimizer = 'adam' , loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy']) <br>
history = model.fit(train_dataset , epochs = 10 , validation_data = test_dataset)

