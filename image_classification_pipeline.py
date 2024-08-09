import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Image Preprocessor Class
class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, data_dir, target_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        self.num_classes = None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        train_generator = self.datagen.flow_from_directory(
            self.data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        validation_generator = self.datagen.flow_from_directory(
            self.data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        self.num_classes = train_generator.num_classes
        return train_generator, validation_generator

# Image Classifier Class
class ImageClassifier(BaseEstimator):
    def __init__(self, base_model=None, learning_rate=0.0001, epochs=10, num_classes=None):
        self.base_model = base_model or ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.num_classes = num_classes

    def fit(self, X, y=None):
        train_generator, validation_generator = X

        self.base_model.trainable = False
        x = Flatten()(self.base_model.output)
        x = Dense(1024, activation='relu')(x)

        if self.num_classes is None:
            self.num_classes = train_generator.num_classes

        predictions = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                           loss='categorical_crossentropy', metrics=['accuracy'])
        
        steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
        validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)

        self.model.fit(
            train_generator,
            validation_data=validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=self.epochs,
            verbose=1
        )
        return self

    def predict(self, X):
        _, validation_generator = X
        validation_generator.reset()
        predictions = self.model.predict(validation_generator)
        return np.argmax(predictions, axis=1)

    def score(self, X, y=None):
        y_pred = self.predict(X)
        _, validation_generator = X
        y_true = validation_generator.classes
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return accuracy

# Example Usage
if __name__ == "__main__":
    # Define the data directory
    data_dir = './data-collection/organized_images'

    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessing', ImagePreprocessor(data_dir=data_dir)),
        ('classification', ImageClassifier(epochs=10))
    ])

    # Fit the pipeline
    pipeline.fit(None)

    # Evaluate the model
    pipeline.score(None)
