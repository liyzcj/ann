import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot  as plt

def load_mnist():
    (x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()
    # Data Normalization
    x_train = x_train / 255
    x_test = x_test  / 255

    # labels to categorical
    y_train = k.utils.to_categorical(y_train, 10)
    y_test = k.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def sap_noise(dataset, rate):
    """
    Add Salt and Peper Noise in Dataset
    dataset: Numpy darray. where the first dimention is the number of image.
    rate: between 0 to 1, the noise level of Salt and Pepper Noise.
    """
    dataset = dataset.copy()
    prob = np.random.rand(*dataset.shape)
    u = np.where(prob > (1 - rate / 2))
    d = np.where(prob < rate / 2)
    dataset[u] = 1
    dataset[d] = 0
    return dataset

def gaussian_noise(dataset, mean, std):
    noise = np.random.normal(loc = mean, scale = std, size = dataset.shape)
    dataset_noise = dataset  +  noise
    dataset_noise = np.clip(dataset_noise, dataset.min(), dataset.max())
    return dataset_noise

def show_mnist(X, title = None, loc = "center"):
    num = X.shape[0]
    imgs = np.moveaxis(X.reshape(-1, num, 28, 28), 1, 2).reshape(-1, num * 28)
    plt.figure()
    if title:
        plt.title(title, loc=loc)
    plt.axis('off')
    plt.imshow(imgs, cmap= "Greys")

def show_mnist_comp(data, model = None, title = None, loc = "center"):

    num = data.shape[0]
    imgs = []
    for l in range(10):
        level = l / 10
        data_noise = sap_noise(data, level)
        if model != None:
            denoise = model.predict(data_noise)
            img = np.moveaxis(denoise.reshape(-1, num, 28, 28), 1, 2).reshape(-1, num * 28)
            imgs.append(img)
            sep = np.zeros((7, num*28))
            imgs.append(sep)
        else:
            img = np.moveaxis(data_noise.reshape(-1, num, 28, 28), 1, 2).reshape(-1, num * 28)
            imgs.append(img)
            sep = np.zeros((7, num*28))
            imgs.append(sep)

    imgs = np.concatenate(imgs)
    plt.figure(dpi=200)
    if title:
        plt.title(title, loc = loc)
    
    plt.axis('off')
    plt.imshow(imgs, cmap="Greys")

if __name__ == "__main__":
    (train, train_l), (test,testl)  = load_mnist()
    plt.figure()
    plt.subplot(121)
    plt.imshow(test[100])
    noisetest = sap_noise(test, 0.9)
    plt.subplot(122)
    plt.imshow(noisetest[100])
    plt.show()
