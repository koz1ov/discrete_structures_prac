# принимает решение о продаже или покупке по последним 50 значениям из последовательности
# запуск: "python3 get_decision.py test.txt"
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import models

if __name__ == '__main__':
    x = eval(open(f'{sys.argv[1]}', 'r').read())
    x = np.array(list(map(float, x)))
    model = models.load_model("result_model")
    next_val = model.predict(x[-50:].reshape(1, -1))[0][0]
    
    if next_val > x[-1]:
        print("BUY")
    elif next_val < x[-1]:
        print("SELL")
    else:
        print("DO NOTHING")
