import pygame as pg
import tensorflow as tf
import numpy as np
from pylsl import resolve_stream, StreamInlet



def main():
    #Boilerplate.
    pg.init()
    pg.font.init()
    fnt = pg.font.SysFont('Arial',36)
    HEIGHT = 500
    WIDTH = 500
    pg.display.set_caption('State Classification')
    screen = pg.display.set_mode((WIDTH,HEIGHT))
    clock = pg.time.Clock()

    print('Establishing LSL connection.')
    streams = resolve_stream('type','EEG')
    inlet = StreamInlet(streams[0])
    print('LSL connection established.')
    model = tf.keras.models.load_model('models/first.h5')

    cont = True
    predictions = [0]*10
    while cont:
        screen.fill((255,255,255))
        delta = clock.tick(30)

        pg.draw.rect(screen,(150,0,0,),(0,0,WIDTH,HEIGHT*.33))
        pg.draw.rect(screen,(0,150,0),(0,HEIGHT*.33,WIDTH,HEIGHT*.33))
        pg.draw.rect(screen,(0,0,150),(0,HEIGHT*.66,WIDTH,HEIGHT*.33))

        #Labels
        screen.blit(fnt.render('Gaming',True,(255,255,255)),dest = (WIDTH*.4,HEIGHT*.1))
        screen.blit(fnt.render('Reading',True,(255,255,255)),dest = (WIDTH*.4,HEIGHT*.45))
        screen.blit(fnt.render('Meditating',True,(255,255,255)),dest = (WIDTH*.4,HEIGHT*.8))

        #Collect data
        data = []
        for i in range(20):
            sample, timestamp = inlet.pull_sample()
            data.append(sample[:60])
        data = np.asarray(data)
        predictions.pop(0)
        predictions.append(model.predict(np.asarray(data).reshape((1,20,60))))

        # Determine the position of the box based on the maximum index from the last 2 seconds of predictions
        sums = sum(predictions).tolist()[0]
        print(sums)
        idx = sums.index(max(sums))
        POS = [HEIGHT*.1,HEIGHT*.4,HEIGHT*.75]

        pg.draw.rect(screen,(0,0,0),(0,POS[idx],WIDTH/8,HEIGHT/10))

        for event in pg.event.get():
            if(event.type == pg.QUIT):
                cont = False
        pg.display.update()


if __name__ == '__main__':
    main()