import threading

def printSomething():
    print("hello")

t1 = threading.Thread(target = printSomething())

t1.start()





t1.join()