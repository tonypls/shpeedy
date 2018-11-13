import cv2
import os
import numpy as np
import createbatches
import initmodel
import train


FRAME_H, FRAME_W = 112, 112
TIMESTEPS = 16

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_folder = 'data/train/'

indices = range(TIMESTEPS-1, len(os.listdir(data_folder + 'images/')), TIMESTEPS)
gen_train = createbatches.BatchGenerator(data_folder, indices, batch_size=4, timesteps=TIMESTEPS)
generator = gen_train.get_gen()
# show(next(generator))

model = initmodel.createmodel(TIMESTEPS, FRAME_H, FRAME_W)

model = train.train(model, TIMESTEPS)

file_name = 'test'  # name without extension of arbitrary dashcam video

video_inp = 'data/' + file_name + '.mp4'
video_out = 'data/' + file_name + '_out.mp4'
label_inp = open('data/' + file_name + '.txt', 'r').readlines() if os.path.exists(
    'data/' + file_name + '.txt') else None
label_out = open('data/' + file_name + '_pred.txt', 'w')

video_reader = cv2.VideoCapture(video_inp)
h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_out, fourcc, 20.0, (w, h))

x_batch_original = np.zeros((TIMESTEPS, h, w, 3))
x_batch = np.zeros((1, TIMESTEPS, FRAME_H, FRAME_W, 3))
frame_counter = 0
frame_counter_all = 0
curr_speed = 0.
acc_error = 0

while (True):
    ret, image = video_reader.read()

    if ret == True:
        if frame_counter_all > -1:  # only start processing from certain frame count
            x_batch_original[frame_counter] = image
            heigh = image.shape[0]
            transformed = np.concatenate([np.arange(heigh / 3), np.arange(heigh * 2 / 3, heigh)])

            image = image[transformed.astype('int64') , :, :]
            image = cv2.resize(image.copy(), (FRAME_H, FRAME_W))
            image = createbatches.normalize(image)

            x_batch[0, frame_counter] = image

            if frame_counter == TIMESTEPS - 1:
                curr_speed = model.predict(x_batch)[0][0]
                frame_counter = -1

                for i in range(TIMESTEPS):
                    image = x_batch_original[i]
                    caption = 'Speed (Predicted): ' + str("{0:.2f}".format(curr_speed))
                    image = cv2.putText(image, caption, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    # write true speed if available
                    if label_inp is not None:
                        true_speed = float(label_inp[frame_counter_all - (TIMESTEPS - 1) + i].strip())
                        caption = 'Speed (Actual): ' + str("{0:.2f}".format(true_speed))
                        image = cv2.putText(image, caption, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                        acc_error += (curr_speed - true_speed) ** 2
                        caption = 'MSE: ' + str(
                            "{0:.2f}".format(acc_error / (frame_counter_all - TIMESTEPS + 1 + i + 1)))
                        image = cv2.putText(image, caption, (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    video_writer.write(np.uint8(image))
                    label_out.write(str(curr_speed) + '\n')

            frame_counter += 1
    else:
        break

    frame_counter_all += 1

video_reader.release()
video_writer.release()

label_out.close()

