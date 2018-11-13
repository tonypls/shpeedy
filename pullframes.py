import cv2

video_file = 'train'

video_inp = 'data/' + video_file + '.mp4'
label_inp = open('data/' + video_file + '.txt').readlines()

video_reader = cv2.VideoCapture(video_inp)
train_image = 'data/' + video_file + '/images/'
train_label = open('data/' + video_file + '/train.txt', 'w')

counter = 0

while(True):
    ret, frame = video_reader.read()

    if ret == True:
        cv2.imwrite(train_image + str(counter).zfill(6) + '.png', frame)
        train_label.write(label_inp[counter])
        counter += 1
    else:
        break

video_reader.release()
train_label.close()