import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. SETUP DATA LOADERS
# We point this to the 'dummy_data' folder you just created
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'dummy_data/train',
    target_size=(224, 224),
    batch_size=2,
    class_mode='categorical'
)

# 2. BUILD THE MODEL (EfficientNetB0)
# 'include_top=False' removes the final 1000-class layer so we can add our own
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base to keep pre-trained knowledge

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(2, activation='softmax') # 2 classes: Class_A and Class_B
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. RUN A TEST EPOCH
print("\n🚀 Starting test training on Mac GPU...")
model.fit(train_generator, epochs=1)
print("\n✅ If you see this, your pipeline and GPU are working perfectly!")