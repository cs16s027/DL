import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Augment:

    def __init__(self, images):
        self.batch = self.augment(images)

    def translate(self, image):
        # Translate
        x, y = np.random.uniform(-10, 10, 2)
        y = 0
        trans_mat = np.float32([[1, 0, x], [0, 1, y]])
        return cv2.warpAffine(image, trans_mat, (28, 28))

    def rotate(self, image):
        theta = np.random.uniform(-30, 30)
        rot_mat = cv2.getRotationMatrix2D((14.0, 14.0), theta, scale = 1.0)
        return cv2.warpAffine(image, rot_mat, (28, 28))
    
    def flipVertical(self, image):
        return cv2.flip(image, 1)

    def dropout(self, image, p = 0.3):
        pixeld = int(p * 784)
        image_ = image.reshape((784))
        drop = np.random.randint(0, 784, pixeld) 
        image_[drop] = 0.0
        return image_.reshape((28, 28))

    def augment(self, images):
        images = np.copy(images).reshape((-1, 28, 28))
        batch = np.zeros_like(images)
        for index in range(images.shape[0]):
            if np.random.uniform(0, 1) < 0.1:
                batch[index, :, :] = images[index, :, :]
            else:
                num = np.random.randint(1, 4)
                augs = np.random.choice([self.translate, self.flipVertical, self.dropout], num, replace = False)
                image_ = np.copy(images[index])
                for aug in augs:
                    image_ = aug(image_)
                batch[index, :, :] = image_
        return batch.reshape((-1, 784))

def writeImages():
    images = np.load('data/train_X.npy')
    for i in range(50):
        image = images[i].reshape((28, 28)) * 256
        cv2.imwrite('data/view/{}.jpg'.format(i), image)

if __name__ == '__main__':

    images = np.load('data/train_X.npy')[:20].reshape((20, 28, 28)) * 256
    for i in range(20):
        cv2.imwrite('data/view/original/{}.jpg'.format(i), images[i])
    augmented = Augment(images).batch
    for i in range(20):
         cv2.imwrite('data/view/augmented/{}.jpg'.format(i), augmented[i])       

