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

