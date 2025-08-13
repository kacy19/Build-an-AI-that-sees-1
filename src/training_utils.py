import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

def compile_and_train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, model_path=None, history_path=None):
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    # Save model
    if model_path:
        model.save(model_path)

    # Save history
    if history_path:
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)

    return history

def plot_training_curves(history, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
