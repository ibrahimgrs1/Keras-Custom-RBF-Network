# %% Gerekli Kütüphanelerin Yüklenmesi
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Dense, Dropout
from tensorflow.keras import backend as K
import warnings

warnings.filterwarnings("ignore")

# %% 1. Veri Seti Hazırlığı
iris = load_iris()
X = iris.data
y = iris.target

# Etiketleri One-Hot Encoding formatına dönüştür (3 sınıf: [1,0,0], [0,1,0], [0,0,1])
label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)

# Veriyi standardize et (0 ortalama, 1 standart sapma) - RBF için kritiktir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi Eğitim ve Test olarak ayır (%20 Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=32)

# %% 2. Özel RBF (Radyal Temelli Fonksiyon) Katmanı Tanımlama
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        # Merkez noktalarını (mu) eğitilebilir ağırlık olarak tanımla
        self.mu = self.add_weight(name="mu",
                                  shape=(int(input_shape[1]), self.units),
                                  initializer="uniform",
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        # Girdi ve merkezler arasındaki farkın karesini (L2 Norm) hesapla
        diff = K.expand_dims(inputs, axis=-1) - self.mu 
        l2 = K.sum(K.pow(diff, 2), axis=1)
        # RBF aktivasyon fonksiyonu: exp(-gamma * mesafe^2)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

# %% 3. Optimize Edilmiş Model Oluşturma
model = Sequential([
    # RBF Katmanı: Nöron sayısını 32'ye çıkardık, gamma'yı hassaslaştırdık
    RBFLayer(units=32, gamma=0.3, input_shape=(X_train.shape[1],)),
    
    # Başarıyı artırmak için ek yoğun katman
    Dense(16, activation='relu'),
    Dropout(0.1), # Ezberlemeyi (overfitting) önlemek için
    
    # Çıktı Katmanı
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# %% 4. Modeli Eğitme
print("Eğitim süreci başlatıldı...")
history = model.fit(X_train, y_train, 
                    epochs=250, # Daha uzun eğitim süresi
                    batch_size=16, 
                    verbose=0, 
                    validation_split=0.1)

# %% 5. Performans Testi ve Görselleştirme
print("\n--- Final Test Sonuçları ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Doğruluğu: %{accuracy*100:.2f}")

# Başarı ve Kayıp Grafiklerini Çizdir
plt.figure(figsize=(12, 4))

# Doğruluk (Accuracy) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı', color='blue')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı', color='orange')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı', color='red')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', color='darkred')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.tight_layout()
plt.show()