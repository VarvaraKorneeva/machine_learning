import pygame
from sklearn import svm


def algorithm(X, y):
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    return clf


if __name__ == '__main__':
    r = 3
    pygame.init()
    length = 600
    screen = pygame.display.set_mode((length, 400), pygame.RESIZABLE)
    screen.fill(color='#FFFFFF')
    pygame.display.update()
    X = []
    y = []
    new_points = []
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.VIDEORESIZE:
                screen.fill(color='#FFFFFF')
                for i, point in enumerate(X):
                    if y[i] == 0:
                        pygame.draw.circle(screen, color='red', center=point, radius=r)
                    if y[i] == 1:
                        pygame.draw.circle(screen, color='blue', center=point, radius=r)

            if event.type == pygame.KEYUP:
                if event.key == 1073742049:
                    points = []
                    screen.fill(color='#FFFFFF')

                if event.key == 13:
                    clf = algorithm(X, y)
                    w = clf.coef_[0]
                    b = clf.intercept_[0]
                    y1 = (- 0 * w[0] - b) / w[1]
                    y2 = (- length * w[0] - b) / w[1]
                    pygame.draw.line(screen, 'black', (0, y1), (length, y2), 3)

                if event.key == 32:
                    clf = algorithm(X, y)
                    res = clf.predict(new_points)

                    for i, point in enumerate(new_points):
                        if res[i] == 0:
                            pygame.draw.circle(screen, color='red', center=point, radius=r)
                        elif res[i] == 1:
                            pygame.draw.circle(screen, color='blue', center=point, radius=r)
                    new_points = []

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_pressed = True
                    coord = event.pos
                    pygame.draw.circle(screen, color='red', center=coord, radius=r)
                    X.append(coord)
                    y.append(0)

                if event.button == 3:
                    coord = event.pos
                    pygame.draw.circle(screen, color='blue', center=coord, radius=r)
                    X.append(coord)
                    y.append(1)

                if event.button == 2:
                    coord = event.pos
                    pygame.draw.circle(screen, color='black', center=coord, radius=r)
                    new_points.append(coord)

            pygame.display.update()
