from scipy.io import wavfile
import cv2
import numpy as np
import pygame
import imutils


def pitchshift(array, n):
    # drops or raises pitch by n semitones by strecthing/ shrinking and then
    # speeding/slowing the signal
    factor = 1/(2**(1.0 * n / 12.0))

    window = 2**13
    height = 2**11 

    phase = np.zeros(window)
    hanning = np.hanning(window)
    newpitch = np.zeros(int(len(array) / factor + window))

    for i in np.arange(0, len(array) - (window + height), height*factor, dtype = int):
        # Two overlapping subarrays
        sub1 = array[i: i + window]
        sub2 = array[i + height: i + window + height]

        #fft into frequency domain
        fft1 = np.fft.fft(hanning * sub1)
        fft2 = np.fft.fft(hanning * sub2)

        # Rephase all frequencies
        phase = (phase + np.angle(fft2/fft1)) % 2*np.pi

        #ifft into time domain
        sub2_new = np.fft.ifft(np.abs(fft2)*np.exp(1j*phase))
        i2 = int(i/factor)
        newpitch[i2: i2 + window] += hanning*sub2_new.real

    # normalize (16bit) 12-bit max to avoid clipping
    newpitch = ((2**(12)) * newpitch/newpitch.max())

    #speed correct sound to proper speed
    indices = np.round(np.arange(0, len(newpitch[window:]), 1/factor))
    indices = indices[indices < len(newpitch[window:])].astype(int)

    return newpitch[indices].astype('int16')

#file for modulation
base_key = 'piano_c.wav'

#keys for pygame
#large for possible genralization to larger key sets
keys = ['0','1','2','3','4','5','6','7','8','9',
        'a','b','c','d','e','f','g','h','i','j',
        'k','l','m','n','o','p','q','r','s','t',
        'u','v','w','x','y','z','!','@','#','$',
        '%','^','&','*','(',')','-','=','+','-']

#threshold for motion detction
THRESHOLD = 1500

#colors for image output
RED = (255,0,0)
BLACK = (255,255,255)
WHITE = (0,0,0)

# height and radius of drawn circles
y = 350
RADIUS = 20

#index of black keys
BLACK_KEYS = (0,2,4,5,7,9,11)

#initialize motion and pressedarray
motion = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
pressed = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # used for only vsually displayed keys, not same as playing array

#open middle C wave file
fps, sound = wavfile.read(base_key)

#generate all sounds from middle C
print('generating all notes')
pianokeys = range(-24,25)
note_sounds = [pitchshift(sound, n) for n in pianokeys]
print('done')
 
#begin camera   
cap = cv2.VideoCapture(0)

#capture reference image
ok, base = cap.read()
print('press q in order to establish reference frame')
while True:
    ok, base = cap.read()
    cv2.imshow('reference frame', base)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

#process refernce image for motion detection. 
base = imutils.resize(base, width=500)
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
base_gray = cv2.GaussianBlur(base_gray, (21, 21), 0)
cv2.destroyAllWindows()

#initialize output window
img = cv2.imread('piano.jpg', 1)
img_reset = img                   
cv2.namedWindow('Live video')

#initialize pygame
pygame.mixer.init(fps, -16, 1, 2048)
notes = map(pygame.sndarray.make_sound, note_sounds)
key_sound = dict(zip(keys, notes))
playing = {k: False for k in keys}
py_screen = pygame.display.set_mode((150, 150))

print('press escape to exit the program')

while True:
    #open outpu window
    cv2.imshow('Live video', img)

    #read one frame
    ok, frame = cap.read()          

    #pre process input frames
    frame = imutils.resize(frame, width=500)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

    #compute delta matrix for motion detection
    delta = cv2.absdiff(base_gray, frame_gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # set input cell and output image width and height form in
    cell_height, width = thresh.shape[:2]
    cell_width = width/7
    im_height, im_width = img.shape[:2]
    im_width = im_width/7
 
    #store motion level for each cell
    #could be pre processed but variable cell length with non uniform key pattern results in
    #multiple condition checks when generalized to larger key sets
    motion[0] = cv2.countNonZero(thresh[0:cell_height,   0: 3*cell_width//4])
    motion[1] = cv2.countNonZero(thresh[0:cell_height,   3*cell_width//4: 5*cell_width//4])
    motion[2] = cv2.countNonZero(thresh[0:cell_height,   5*cell_width//4: 7*cell_width//4])
    motion[3] = cv2.countNonZero(thresh[0:cell_height,   7*cell_width//4: 9*cell_width//4])
    motion[4] = cv2.countNonZero(thresh[0:cell_height,   9*cell_width//4: 12*cell_width//4])
    motion[5] = cv2.countNonZero(thresh[0:cell_height,   12*cell_width//4: 15*cell_width//4])
    motion[6] = cv2.countNonZero(thresh[0:cell_height,   15*cell_width//4: 17*cell_width//4])
    motion[7] = cv2.countNonZero(thresh[0:cell_height,   17*cell_width//4: 19*cell_width//4])
    motion[8] = cv2.countNonZero(thresh[0:cell_height,   19*cell_width//4: 21*cell_width//4])
    motion[9] = cv2.countNonZero(thresh[0:cell_height,   21*cell_width//4: 23*cell_width//4])
    motion[10] = cv2.countNonZero(thresh[0:cell_height,  23*cell_width//4: 25*cell_width//4])
    motion[11] = cv2.countNonZero(thresh[0:cell_height,  25*cell_width//4: width])

    #process motion 
    for i in range(motion.shape[0]):
        #key pressed
        if (motion[i] > THRESHOLD) and (pressed[i] == 0):
            pygame.event.post(pygame.event.Event(2,{'unicode':0,'key':ord(keys[19+i]),'mod':0}))
            cv2.circle(img,((i+1+(1 if (i > 4) else 0))*im_width/2,y),RADIUS,RED,-1)
            pressed[i] = 1
        #key released
        elif(motion[i] < THRESHOLD) and (pressed[i] == 1):
            pygame.event.post(pygame.event.Event(3,{'key':ord(keys[19+i]),'mod':0}))
            cv2.circle(img,((i+1+(1 if (i > 4) else 0))*im_width/2,y),RADIUS,(BLACK if (i in BLACK_KEYS) else WHITE),-1)
            pressed[i] = 0

    #process all pygame events
    event = pygame.event.poll()
    while event.type != pygame.NOEVENT:
        #filter out all non keypress events
        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
            key = pygame.key.name(event.key)

        #key pressed
        if event.type == pygame.KEYDOWN:
            if (key in key_sound.keys()) and (not playing[key]):
                key_sound[key].play(fade_ms=50)
                playing[key] = True

            #game quit
            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                print ( "Thank you for playing music with us")
                exit()

        #keyreleased
        elif event.type == pygame.KEYUP and key in key_sound.keys():
            # Stops with 50ms fadeout
            key_sound[key].fadeout(500)
            playing[key] = False
        event = pygame.event.poll()

   