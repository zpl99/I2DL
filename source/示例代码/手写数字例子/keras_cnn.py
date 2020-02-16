from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta


def getModel():
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1]))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                 optimizer=Adadelta(),
                 metrics=['accuracy'])
    return model