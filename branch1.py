import threading
import time

#-------------MAIN THREAD---------------------
#YOLO recognizes the license plate in the frame
#crops the area around the license plate
#####saves the crop in a directory------------------------> [START THREAD]
#display the frame and the results from the extraction if any
#-----------------END-------------------------


#--------------------------THREAD---------------------
#run OCR on crop
#check the extracted license if it fits format
#return result to main thread
#------------------------ACTIVATE AGAIN IF CROP IS THERE-----------------


def extractLicense(frames):

    while True:
        global stop_threads
        time.sleep(1)
        print(f"thread works at frame {frames}")
        if stop_threads:
            break


licensePlate = ''
count = 0
global frames
frames = 10
it = 4



#running the video
while frames != 0:
    stop_threads = False
    print(stop_threads)
    t1 = threading.Thread(target=extractLicense, args=(frames,))
    t1.start()



    print(f'Main Thread - frame: {-1 * (frames - 10)}')
    licensePlate = "license plate: " + str(count)

    time.sleep(4)

    frames -= 1
    count +=1
    stop_threads = True

t1.join()



