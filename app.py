{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7cebd7fc",
   "metadata": {},
   "source": [
    "# Md Serajun Nabi\n",
    "AIU20092069, Deep Learning_Lab 07"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "15af9b7a",
   "metadata": {},
   "source": [
    "# Autoencoder Algorithm"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "624564b2",
   "metadata": {},
   "source": [
    "# Simple Autoencoder\n",
    "We will build simple Autoencoder algorithm, with a single fully-connected neural layer as encoder and as decoder:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0691e343",
   "metadata": {},
   "outputs": [],
   "source": [
    "import keras\n",
    "from keras import layers\n",
    "\n",
    "encoding_dim = 12\n",
    "\n",
    "input_img = keras.Input(shape=(784,))\n",
    "\n",
    "encoded = layers.Dense(encoding_dim, activation='relu')(input_img)\n",
    "\n",
    "decoded = layers.Dense(784, activation='sigmoid')(encoded)\n",
    "\n",
    "autoencoder = keras.Model(input_img, decoded)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3be052c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "encoder = keras.Model(input_img, encoded)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "057cf0f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "encoded_input = keras.Input(shape=(encoding_dim,))\n",
    "\n",
    "decoder_layer = autoencoder.layers[-1]\n",
    "\n",
    "decoder = keras.Model(encoded_input, decoder_layer(encoded_input))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b20b0893",
   "metadata": {},
   "outputs": [],
   "source": [
    "autoencoder.compile(optimizer='adam', loss='binary_crossentropy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a555d589",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
      "11490434/11490434 [==============================] - 3s 0us/step\n"
     ]
    }
   ],
   "source": [
    "from keras.datasets import mnist\n",
    "import numpy as np\n",
    "(x_train, _), (x_test, _) = mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "daf447aa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(60000, 784)\n",
      "(10000, 784)\n"
     ]
    }
   ],
   "source": [
    "x_train = x_train.astype('float32') / 255.\n",
    "x_test = x_test.astype('float32') / 255.\n",
    "x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))\n",
    "x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))\n",
    "print(x_train.shape)\n",
    "print(x_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ab802ff9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.4847 - val_loss: 0.2059\n",
      "Epoch 2/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0972 - val_loss: 0.0420\n",
      "Epoch 3/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0273 - val_loss: 0.0177\n",
      "Epoch 4/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0137 - val_loss: 0.0107\n",
      "Epoch 5/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0091 - val_loss: 0.0078\n",
      "Epoch 6/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0071 - val_loss: 0.0064\n",
      "Epoch 7/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0060 - val_loss: 0.0057\n",
      "Epoch 8/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0054 - val_loss: 0.0052\n",
      "Epoch 9/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0051 - val_loss: 0.0050\n",
      "Epoch 10/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0048 - val_loss: 0.0048\n",
      "Epoch 11/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0047 - val_loss: 0.0046\n",
      "Epoch 12/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0046 - val_loss: 0.0045\n",
      "Epoch 13/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0045 - val_loss: 0.0045\n",
      "Epoch 14/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0044 - val_loss: 0.0044\n",
      "Epoch 15/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0044 - val_loss: 0.0044\n",
      "Epoch 16/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0043 - val_loss: 0.0043\n",
      "Epoch 17/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0043 - val_loss: 0.0043\n",
      "Epoch 18/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0043 - val_loss: 0.0043\n",
      "Epoch 19/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0042 - val_loss: 0.0043\n",
      "Epoch 20/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0042 - val_loss: 0.0043\n",
      "Epoch 21/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0042 - val_loss: 0.0042\n",
      "Epoch 22/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0042 - val_loss: 0.0042\n",
      "Epoch 23/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0042 - val_loss: 0.0042\n",
      "Epoch 24/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0042 - val_loss: 0.0042\n",
      "Epoch 25/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0042 - val_loss: 0.0042\n",
      "Epoch 26/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0041 - val_loss: 0.0042\n",
      "Epoch 27/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0041 - val_loss: 0.0042\n",
      "Epoch 28/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0041 - val_loss: 0.0042\n",
      "Epoch 29/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0041 - val_loss: 0.0042\n",
      "Epoch 30/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0041 - val_loss: 0.0042\n",
      "Epoch 31/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0041 - val_loss: 0.0041\n",
      "Epoch 32/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0041 - val_loss: 0.0041\n",
      "Epoch 33/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0041 - val_loss: 0.0041\n",
      "Epoch 34/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0041 - val_loss: 0.0041\n",
      "Epoch 35/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0041 - val_loss: 0.0041\n",
      "Epoch 36/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0041 - val_loss: 0.0041\n",
      "Epoch 37/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0041 - val_loss: 0.0041\n",
      "Epoch 38/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0041\n",
      "Epoch 39/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0040 - val_loss: 0.0041\n",
      "Epoch 40/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0041\n",
      "Epoch 41/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0041\n",
      "Epoch 42/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0041\n",
      "Epoch 43/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0041\n",
      "Epoch 44/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0041\n",
      "Epoch 45/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0040\n",
      "Epoch 46/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0040\n",
      "Epoch 47/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0040\n",
      "Epoch 48/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0040\n",
      "Epoch 49/50\n",
      "235/235 [==============================] - 1s 3ms/step - loss: 0.0040 - val_loss: 0.0040\n",
      "Epoch 50/50\n",
      "235/235 [==============================] - 1s 2ms/step - loss: 0.0040 - val_loss: 0.0040\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x21d2e484d60>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "autoencoder.fit(x_train, x_train,\n",
    "               epochs=50,\n",
    "               batch_size=256,\n",
    "               shuffle=True,\n",
    "               validation_data=(x_test, x_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "3b6c30bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "313/313 [==============================] - 0s 495us/step\n",
      "313/313 [==============================] - 0s 509us/step\n"
     ]
    }
   ],
   "source": [
    "encoded_imgs = encoder.predict(x_test)\n",
    "decoded_imgs = decoder.predict(encoded_imgs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "793253e8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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HEwAAAAAAADVKH49QosXL85iH/jAB7LYzJnt86Ozzjor66PI0+T0la98JYs96EEPGvi6j3/841nszDPP7GNIsFO3v/3tB/aJCgczee2yS36b0LXY9Q9/+MMsdsopp7TaaSG8iYiKX7/5zW/OYu94xzta7T333DPrE+X11772tVb78ssvv7lDJPDYxz42i6XvSXSvNA6iYvGnnnpqFtu2bVur/YY3vCHro5j61HbPe96zKJaKCiv+5je/6WNIjImHPexhrfZ3vvOdrM/q1auzWFSYs6u0qPH97ne/rM897nGPomt94Qtf6GNIDNHs2bNb7ahQ+zvf+c6ia23evLnV/shHPpL1idb5I444YuC1o0LI41Acfrp71KMe1Wq//OUvz/osWbIki93nPvdptdesWdPruJheonXqjDPOyGJpseu//vWvWZ/HPOYxWewXv/hF98El0iLWBx98cNYn+rzvW9/6VhabP3/+wJ8XFfg+99xzW+3ovqIm35gAAAAAAACqcTABAAAAAABU42ACAAAAAACoRo2JETrttNOy2KJFi7LYqlWrWu0//vGPQxsTw3PAAQdkseh5wulzPqNnrkfPo16/fv0ERgdt0bODn/KUp2SxX//61632d7/73aGNicnjwgsvzGJPfepTs1ifNSVKpHUhmiavAXDcccfVGs60N2/evCxW8tzyPp+l3qdnPvOZWSyqyXLxxRe32t///veHNibGU9d5Zlxzn8He/e53Z7H73//+WezAAw9ste973/tmfaLnRT/iEY+YwOhu+vpRzYHIn//85yz2yle+spcxMTxPeMITBvZJa580TVwrscRd73rXTq+74IILspj97+iV1EdK94tN0zRXX331MIbDNJXWbWiavKZbZOvWrVns7ne/exY7+eSTs9itb33rgdfftGlTFrvNbW5zk+2miffI++2338CfF1m2bFkWSz9PHHVtO9+YAAAAAAAAqnEwAQAAAAAAVONgAgAAAAAAqMbBBAAAAAAAUI3i15Xc6173ymIvf/nLi177qEc9qtX+3e9+18eQqOyLX/xiFlu4cOHA133iE5/IYpdffnkvY4KdOf7447PYggULsth5553Xam/evHloY2I8zJw5+P80REXDxkFUMDT9fUp+v6Zpmte+9rWt9pOe9KTO45quZs+encVucYtbZLFPf/rTNYYzYUceeWRRP/dx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/vWvVxgJ4yQq7jZz5uD/01BSTLNpmuZDH/pQq33ggQcWvS4dw/bt24teV+LEE0/s7VoMx29+85ui2DD9+c9/7vzaY445ptX+3e9+N9HhMEbuec97ZrGSefMrX/nKEEYDNy1arzds2NBqv/3tb681HNipz33uc1ksKn79+Mc/vtU+/fTTsz5nnXVWfwOjF9/73veK+h177LFZLC1+vXXr1qzPRz7ykSz27//+7632C17wgqzPE5/4xKJxMbXd7W53y2Lp2jhnzpyia61fv77Vfvazn531ueGGG27G6CY/35gAAAAAAACqcTABAAAAAABU42ACAAAAAACoRo2JAdJaEeedd17W5/DDD89il19+eav96le/ut+BMSX89re/Hdq1P//5z2exa665Jovtt99+WSx9NucoXHvtta32G9/4xhGNZOq5973vncX233//EYyEcXf22Wdnsbe85S0DX/eNb3wji5XUgehaK2IiNSbOOeeczq9l+orqr0SxiJoSU1tUjy513XXXZbF3v/vdwxgO/H/Rc6yjfcDy5ctb7V/96ldDGxOUiu71onvSRz7yka32v/3bv2V9PvOZz2SxSy+9dAKjo5bvfOc7WSz9nGCXXfKPOZ/xjGdksaOOOqrVvt/97td5XFdffXXn1zL+ohqEc+fOHfi6tGZT0+S1cX760592H9gU4RsTAAAAAABANQ4mAAAAAACAahxMAAAAAAAA1TiYAAAAAAAAqlH8eoAjjzyy1b7LXe5S9LoXvehFrXZaDJup5Vvf+larnRbdGoXHPvaxvV1r69atWayk2OzXvva1LHbhhRcW/cwf//jHRf24+U466aQsNmvWrFb717/+ddbnRz/60dDGxHj60pe+lMXOOOOMVnvRokW1hrNTK1asyGIXX3xxFnvmM5+Zxa655pqhjImpbceOHUUxpp8HP/jBA/ssWbIki61Zs2YYw4H/Lyp+Hc1b3/zmNwdeKyr6OX/+/CwW5Tr05Te/+U0We81rXtNqv/Wtb836vOlNb8piT3rSk1rtTZs2TWxwDEV0f/+5z32u1X7c4x5XdK373//+A/ts27Yti0Vz5Mtf/vKin8n4i9a3l770pZ2u9clPfjKL/eAHP+h0ranMNyYAAAAAAIBqHEwAAAAAAADVOJgAAAAAAACqcTABAAAAAABUo/j13zn00EOz2He+852Br0sLgTZN03zjG9/oZUxMDo9+9KNb7ag4zq677trp2re73e2y2OMf//hO1/rwhz+cxa644oqBr/viF7+YxS655JJOY6CuPffcM4udcMIJA1/3hS98IYtFxb+Y2q688sosdsopp7Taj3rUo7I+z3/+84c1pNAb3/jGLPb+97+/6hiYXnbfffeifopnTm3Rvd2RRx458HWbN2/OYlu2bOllTDBR6f3eqaeemvV54QtfmMV+//vfZ7EnP/nJ/Q0MCnz84x9vtZ/1rGdlfdK9e9M0zVlnndVq//a3v+13YPQiuq96wQte0GrPmTMn63PXu941iy1evLjVjj4XOffcc7PYa1/72pseJJNGlCt/+MMfsljJZ3nRnJHmJjHfmAAAAAAAAKpxMAEAAAAAAFTjYAIAAAAAAKhGjYm/88xnPjOLHXLIIQNf98Mf/jCL7dixo5cxMTm95S1vGer1n/jEJw71+kwd0TOrV61alcW+9rWvtdrvfve7hzYmJrcf/ehHN9lumrg+U7TGnnjiia12modN0zQf+tCHstiMGTNa7ehZoDBMT3nKU7LY6tWrs9jrX//6CqNhVLZv357FLrzwwix2zDHHtNqXXXbZ0MYEE/X0pz+91X7a056W9fmP//iPLGa+YxysWLGi1T7++OOzPlEtgZe97GWtdlRbhfG0bNmyVjvdXzRN0zzpSU/KYve4xz1a7de97nVZn+XLl09wdIyzBzzgAVnsoIMOymIln+9GtZeimmLkfGMCAAAAAACoxsEEAAAAAABQjYMJAAAAAACgGgcTAAAAAABANdO2+PW9733vLPbc5z53BCMBGJ6o+PU973nPEYyE6eS8884risFk9T//8z9Z7B3veEcW+/73v19jOIzItm3bstirXvWqLJYWTfzlL385tDHBzpx++ulZ7KyzzspiP/rRj1rts88+O+uzatWqLHbjjTdOYHQwHEuWLMli559/fhZ7xCMe0Wrf9ra3zfr84Q9/6G9gVHXuuecWxZheXv/612exkkLXTdM0b33rW1tt9/zd+cYEAAAAAABQjYMJAAAAAACgGgcTAAAAAABANQ4mAAAAAACAaqZt8ev73Oc+WWzOnDkDX3f55ZdnsfXr1/cyJgAAxt+JJ5446iEwppYuXZrFnvrUp45gJND2k5/8JIs94AEPGMFIYLROPvnkLHbRRRe12kcddVTWR/FrmFoWLFiQ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Z9xN55xrmvK8i55hmiqZ60rzLprr0jzrOtetWbMmi0V5F+VnSd5Fr0vzbro/d72rKF/7zLuSfn3Od9H802felayx8q4tneui5xBHOZc+szXKpWheq51zw57r0ufGlubcdJf+mxv2XNd1jY2unypdY4e5n7DGlulrH1s63w1zHzvse7uSPUXp86+nYt513U+U1nEa5j52MuecfWzdfWyUF9E92mTOu+j69rHdlNzb9TnfjcO9XZ/z3bjd2/nGBAAAAAAAUI2DCQAAAAAAoBoHEwAAAAAAQDUOJgAAAAAAgGpGUvw6KpgRFdtIC35EBUCiYjtpcZ20aGLTxIV0Fi1alMX23XffVjsqhBIVlkoLmEQFR6699tosdsUVV2Sxq666qtVevnz5wJ/XNN0L6UxFUc5FuZPm2A033JD1if6Oac6lRXWaJs65xYsXD+xXmnNpMZw+c27ZsmVZHzk3WGnepcXVSguazpo1q9WO5rqoCFM016X9onztc677y1/+ksXSvFuxYkXWp88CiVNV1zU2mu9K1thovovmrWHOd1ExMGvsaEV/j3RuK825dK4rXWOjuS7tFxWzK8m59evXZ326znXRGrt69eosVlKYc7oXSCxZY6O8i/5uJfuJaI2N5rq+1thorrvmmmuy2J///OcsVrLGRte3xrb1uacoybuJ7CnSvIvmu3R+bZrh7mOjvCtZY0vnu6mo635iFGusz06mjnHYx07mvLOP7U+f891kubeL8m4q39v5xgQAAAAAAFCNgwkAAAAAAKAaBxMAAAAAAEA1DiYAAAAAAIBqxrr49caNG1vtqBDgypUrs1hadCQqQhKNIS2E0jR5UZ6oqEo6zqZpmqVLl7baUYGcSy65JItdeumlWezKK69sta+77rqsT1SEMR1rVPxluhTSiX7PqDBTXzkXFcyJij5FxXDSPCzNubT4YVRw849//GNRLC2ic/3112d9ouJNcq6tz7kueg/SIoaleRcVOkzzbvPmzQPH2TRleVc61y1ZsqTVjua6KO/SApPyrmy+27BhQ6tdmndpQbko76IxlORd1/kuWmO7znelxerSsY66aNgoRfPMMHMuuq8rnevSWDp/NE1ZzqX3Zk3T731dlHPpvDydc65phr/Glsx1w15j0/1EVPiwNO/Sua50P1Gyxk73grC157vSNbbrPrZkje16b9d1jZ3O93bj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\n",
      "text/plain": [
       "<Figure size 2000x400 with 20 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "n = 10\n",
    "plt.figure(figsize=(20, 4))\n",
    "for i in range(n):\n",
    "    ax = plt.subplot(2, n, i + 1)\n",
    "    plt.imshow(x_test[i].reshape(28, 28))\n",
    "    plt.gray()\n",
    "    ax.get_xaxis().set_visible(False)\n",
    "    ax.get_yaxis().set_visible(False)\n",
    "    \n",
    "    ax = plt.subplot(2, n, i + 1 + n)\n",
    "    plt.imshow(decoded_imgs[i].reshape(28, 28))\n",
    "    plt.gray()\n",
    "    ax.get_xaxis().set_visible(False)\n",
    "    ax.get_yaxis().set_visible(False)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8bf05d37",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
