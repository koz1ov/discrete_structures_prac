# предсказывает следующие 5 значений из выходной последовательности и рисует график
# запуск : "python3 pred_next_values.py test.txt"
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import models

if __name__ == '__main__':
    x = eval(open(f'{sys.argv[1]}', 'r').read())
    x = np.array(list(map(float, x)))
    fig, ax = plt.subplots()
    model = models.load_model("result_model")
    x_pred = np.array([])
    for i in range(5):
        next_val = model.predict(np.concatenate((x[-50 + i:], x_pred)).reshape(1, -1))[0][0]
        x_pred = np.append(x_pred, next_val)
   
    ax.plot(np.arange(1, 6), x[-5:], color='green')
    ax.plot(np.arange(6, 11), x_pred, color='red')
    plt.show()
