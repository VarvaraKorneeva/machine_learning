import pygame
from keras.models import load_model
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN


def predict(model, img):
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    res = model.predict([img])[0]
    return np.argmax(res), '{:f}'.format(max(res))


def create_dbscan(points):
    eps = 10
    minPts = 5
    clustering = DBSCAN(eps=eps, min_samples=minPts)
    clustering.fit(points)
    return clustering.labels_


def cat_image(image, points):
    x_max = max([p[0] for p in points])
    x_min = min([p[0] for p in points])
    y_max = max([p[1] for p in points])
    y_min = min([p[1] for p in points])
    z = (x_max - x_min) - (y_max - y_min)
    if z < 0:
        b_x = 10 + round(abs(z) / 2)
        b_y = 10
    else:
        b_y = 10 + round(z / 2)
        b_x = 10
    im_crop = image.crop((x_min-b_x, y_min-b_y, x_max+b_x, y_max+b_y))
    im_crop.save('num.jpeg')
    return im_crop


def split_picture(points, labels):
    res = []
    pont = {}
    for k in set(labels):
        pont[k] = []
    for i in range(len(points)):
        pont[labels[i]].append(points[i])
    for lab in set(labels):
        screen.fill(color=SCREEN_COLOR)
        for point in pont[lab]:
            pygame.draw.circle(screen, color=POINT_COLOR, center=point, radius=r)
        pygame.image.save(screen, 'num.jpeg')
        im = Image.open('num.jpeg')
        image = cat_image(im, pont[lab])
        res.append(predict(model, image)[0])
    screen.fill(color=SCREEN_COLOR)
    for point in points:
        pygame.draw.circle(screen, color=POINT_COLOR, center=point, radius=r)

    return res


if __name__ == '__main__':
    SCREEN_COLOR = '#000000'
    POINT_COLOR = '#FFFFFF'
    model = load_model('option.h5')
    r = 2
    pygame.init()
    screen = pygame.display.set_mode((500, 300), pygame.RESIZABLE)
    screen.fill(color=SCREEN_COLOR)
    pygame.display.update()
    is_pressed = False
    points = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.VIDEORESIZE:
                screen.fill(color=SCREEN_COLOR)
                for point in points:
                    pygame.draw.circle(screen, color=POINT_COLOR, center=point, radius=r)
            if event.type == pygame.KEYUP:
                if event.key == 13:
                    labels = create_dbscan(points)
                    res = split_picture(points, labels)
                    print(res)
                    n = ''
                    for el in res:
                        if el == 10:
                            n += '-'
                        else:
                            n += str(el)
                    print(n)
                if event.key == 9:
                    points = []
                    screen.fill(color=SCREEN_COLOR)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_pressed = True
                    coord = event.pos
                    pygame.draw.circle(screen, color=POINT_COLOR, center=coord, radius=r)
                    points.append(coord)

            if event.type == pygame.MOUSEBUTTONUP:
                is_pressed = False

            if is_pressed:
                coord = event.pos
                pygame.draw.circle(screen, color=POINT_COLOR, center=coord, radius=r)
                points.append(coord)

            pygame.display.update()
