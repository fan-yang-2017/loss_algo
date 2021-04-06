import time
import tensorflow as tf
import tensorflow_addons as tfa


total = 0
for i in range(100):
    start=time.time()
    gl = tfa.losses.GIoULoss()
    boxes1 = [[4.0, 3.0, 7.0, 5.0],[5.0, 6.0, 10.0, 7.0]]#, [5.0, 6.0, 10.0, 7.0]
    boxes2 = [[3.0, 4.0, 6.0, 8.0],[14.0, 14.0, 15.0, 15.0]]#, [14.0, 14.0, 15.0, 15.0]
    loss = gl(boxes1, boxes2)
    end=time.time()
    print(loss)
    print(end-start)
    total += (end-start)
print(total/100)