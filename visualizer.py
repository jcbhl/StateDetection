import pygame as pg
from operator import add
import tensorflow as tf
import queue
import numpy as np
from pylsl import resolve_stream, StreamInlet




#Boilerplate.
pg.init()
pg.font.init()
fnt = pg.font.SysFont('Arial',36)
HEIGHT = 500
WIDTH = 500
pg.display.set_caption('State Classification Widget')
screen = pg.display.set_mode((WIDTH,HEIGHT))
clock = pg.time.Clock()

# print('Establishing LSL connection.')
# streams = resolve_stream('type','EEG')
# inlet = StreamInlet(streams[0])
# print('LSL connection established.')
model = tf.keras.models.load_model('models/latest.h5')

cont = True
predictions = []
for i in range(10):
    predictions.append(0)
while cont:
    screen.fill((255,255,255))
    delta = clock.tick(30)

    pg.draw.rect(screen,(130,0,0),(WIDTH/2,0,WIDTH/2,HEIGHT/2)) #Top right
    pg.draw.rect(screen,(0,130,0),(0,0,WIDTH/2,HEIGHT/2)) #Top left
    pg.draw.rect(screen,(0,0,130),(0,HEIGHT/2,WIDTH/2,HEIGHT/2)) #Bottom left
    pg.draw.rect(screen,(130,130,130),(WIDTH/2,HEIGHT/2,WIDTH/2,HEIGHT/2)) #Bottom right

    #Labels
    screen.blit(fnt.render('Label 1',True,(255,255,255)),dest = (WIDTH*.2,HEIGHT*.2)) #Top left
    screen.blit(fnt.render('Label 2',True,(255,255,255)),dest = (WIDTH*.2,HEIGHT*.7)) #Bottom left
    screen.blit(fnt.render('Label 3',True,(255,255,255)),dest = (WIDTH*.7,HEIGHT*.2)) #TOp right
    screen.blit(fnt.render('Label 4',True,(255,255,255)),dest = (WIDTH*.7,HEIGHT*.7)) #Bottom right

    #Collect data
    data = []
    for i in range(20):
        sample, timestamp = inlet.pull_sample()
        data.append(sample[:60])
    data = np.asarray(data)
    predictions.pop(0)
    predictions.append(model.predict(data))

    sums = sum(predictions)
    #TODO: change arrow location based on max index in positions





    for event in pg.event.get():
        if(event.type == pg.QUIT):
            cont = False
    pg.display.update()
