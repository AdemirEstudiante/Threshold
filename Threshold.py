import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Tareas/Tarea05/mapaRadom.png')
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# quitamos el ruido
kernel = np.ones((2,2),np.uint8)

closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# seguro área de fondo
sure_bg = cv2.dilate(closing,kernel,iterations=3)

# Encontrar un área de primer plano segura
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
# encontrar region desconocida
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Etiquetado de marcadores
ret, markers = cv2.connectedComponents(sure_fg)
# Agregue uno a todas las etiquetas para que el fondo seguro no sea 0, sino 1
markers = markers+1

# Ahora, marque la región de desconocido con cero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
plt.subplot(211),plt.imshow(rgb_img)
plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(thresh, 'gray')
#plt.imsave(r'thresh.png',thresh)
plt.title("Threshold"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()